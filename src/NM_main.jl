#using Base: Float64
using Plots
using LinearAlgebra
using Optim
using Statistics
using LineSearches
using JLD2
using Parameters
using DataFrames
using DSP

#these global structures are mainly used for debugging, scheduled to be deleted
global CurrentRun = Array{Float64, 1}[]
global LastPos = Float64[] #store most recent simulation final position
global Lastnbreak = 0::Int
global LastMyosinTraj = Vector{Float64}[]

const Version = 68

#specify time-step
const Δt = 0.05 

## Specify channel geometry, normalise so that the minimum channel width is 0.5 at x=0
const R = 14.0 #pillar radius
const W = 0.1 #thickness of membrane and NE
ct = [0.0, R + 1.25]::Vector{Float64} #centre of top pillar

const N, M = 15, 30 #number of points in nucleus/membrane 

#forces in pN, lengths in μm, times in minutes
@with_kw struct CellParameters
    V0n::Float64 #initial nuclear volume
    V0cyt::Float64 #initial cytoplasm volume
    σMT::Float64 = 5000.0 #Initial total constractility
    α::Float64 = 0.1 #control flow of cytoplasm between chambers
    β::Float64 = 0.1 #control relocalisation of myosin in membrane
    ct::Vector{Float64} = ct #centre of top pillar
    R::Float64 = R #pillar radius
    Δt::Float64 = Δt #time step
    BurstAddition::Float64 = 0.5 #strength of additional contractility released at rhoburst
    ρN::Float64 = 1000.0 #nuclear volume preservation
    ρmem::Float64 = 3000.0 # cell volume preservation
    σN::Float64 = 5000.0 #nuclear contractility
    ηC::Float64 = 0.0 #nucleus resistance to changes in curvature
    ηCM::Float64 = 0.0 #bending rigidity
    VMT::Float64 = 30.0 #microtubule volume
    η::Float64 = 0.5 #friction between nucleus and cytoplasm
    κsteric::Float64 = 10000.0 #steric potential between membrane and pillar
    κintersect::Float64 = 10000.0 #steric potential between nucleus and membrane
    ξ::Float64 = 100000.0 #Constitutive elasticity
    μ::Float64 = 0.1 #tendency for nucleus to centre in cell
    σML0::Float64 = σMT/2 #initial contractility on left
    σS0::Float64 = 10.0 #initial myosin contractility in storage 
    ForceM::Float64 = 5000.0 #external forcing on membrane COM
    ηM::Float64 = 10000.0 #Friction between cell and environment
    ξmem::Float64 = 50000.0 #(linear) elasticity
    Maxmyosin::Float64 = 0.9 #maximum fraction of myosin on left
    W::Float64 = W #thickness of membrane and NE
    PCutoff::Float64 = Inf #pressure value at which the microtubule cage will collapse
    τ::Float64 = 000.0 #Strength of membrane cage in left compartment
    βbar::Float64 = 0.1 #difussive exchange parameter
    VCollapse::Float64 = VMT #size of MT cushion after collapse
    σMBG::Float64 = 0. #background membrane contractility
    Force::Float64 = 1000. #force reassigned to nucleus when entering contraction
    NewPressure::Bool = true #use new pressure calculation, never don't do this really
    ChangeForce::Bool = true #reallocate forcing to the nucleus on occlusion
    MTdecay::Float64 = 0.2 #rate at which MT cushion dissacociates after RB
end

if !isdefined(Main, :Runs) #check if this is the first time this code has been compiled if so, set up global storage arrays
    global Runs = Array{Array{Float64, 1}, 1}[]
    global NMs = Tuple{Int64 ,Int64}[]
    global Vars = Array{@NamedTuple{V0cytLeft::Float64, σML::Float64, RhoBurst::Bool, σSL::Float64}}[]
    global Consts = CellParameters[]

end

"""
    Compare two parameter objects and print the differences between them
"""
function Compare(c1::CellParameters, c2::CellParameters)
    for fn in fieldnames(CellParameters)
        c1f, c2f = getfield(c1, fn), getfield(c2, fn)
        (c1f != c2f) && println("c1.$fn = $c1f : c2.$fn = $c2f")
    end
    
end
function Compare(R1::Int, R2::Int) 
    R1 = RImod(R1)
    R2 = RImod(R2)
    Compare(Consts[R1], Consts[R2])
    NMs[R1] != NMs[R2] && println("NM1 = $(NMs[R1]) : NM2 = $(NMs[R2])")
    length(Runs[R1]) !== length(Runs[R2]) && println("T1 = $(length(Runs[R1])) : T2 = $(length(Runs[R2]))")
    return
end

Compare(R1::Int, c1::CellParameters) = Compare(Consts[RImod(R1)], c1)
Compare(c1::CellParameters, R1::Int) = Compare(c1, Consts[RImod(R1)])

## Initialise geometry
TopPillarFunction(x) = (x >= ct[1]-R && x<= ct[1]+R) ? ct[2] - sqrt(R^2-x^2) : NaN
TopInnerPillarFunction(x) = (x >= ct[1]-(R-W) && x<= ct[1]+R-W) ? ct[2] - sqrt((R-W)^2-x^2) : NaN

TopPillarFunction(x::Float64, ct1::Float64, ct2::Float64, R::Float64)::Float64 = (x >= ct1-R && x<= ct1+R) ? ct2 - sqrt(R^2-x^2) : NaN
TopInnerPillarFunction(x::Float64, ct1::Float64, ct2::Float64, R::Float64)::Float64 = (x >= ct1-(R-W) && x<= ct1+R-W) ? ct2 - sqrt((R-W)^2-x^2) : NaN

const zl = (-20.0)#left circle centre 
const Rl = 4.0 #left nucleus Radius

const Rlm = 15.0 #left membrane radius

include("NM_Auxillary.jl")
include("NM_PlotTools.jl")
include("NM_EnergyFunctional.jl")

## Initialise as an ellipse
InitialPosition(x::Number) = (zl-Rl <= x && x <= zl+Rl) ? Rl*sqrt(1-((x-zl)/(Rl))^2) : 0.0
InitialPositionMembrane(x::Number) = (zl-Rlm <= x && x <= zl+Rlm) ? Rlm*sqrt(1-((x-zl)/(Rlm))^2) : 0.0

##Define discretisation of left, right segments
X = EvenSpacing(zl-Rl, zl+Rl, InitialPosition, N)
Xm = EvenSpacing(zl-Rlm, zl+Rlm, InitialPositionMembrane, M)

Y, Ym = InitialPosition.(X), InitialPositionMembrane.(Xm)

x0vec = [X; InitialPosition.(X); Xm; InitialPositionMembrane.(Xm)]

ℓ0 = norm([X[2]-X[1], Y[2]])
 
const V0n = SimpsonsQuadrature(InitialPosition, zl-Rl+0.01, zl+Rl-0.01)
const V0m = SimpsonsQuadrature(InitialPositionMembrane, zl-Rlm+0.01, zl+Rl-0.01)
const V0cyt = (V0m - V0n)
const V0cytLeft = (SimpsonsQuadrature(InitialPositionMembrane, zl-Rlm+0.01, 0.0) - SimpsonsQuadrature(InitialPosition, zl-Rl+0.01, 0.0))
const V0cytRight = (V0cyt - V0cytLeft)

COM0 = getCOM(x0vec[1:2N])
COM0m = getCOM(Xm, InitialPositionMembrane.(Xm))

Params = CellParameters(V0n=V0n, V0cyt=V0cyt)



"""
 Run the simulation with initial condition specified by x0vec=[X; Y; Xm; Ym]
 Optional arguments:
    T is total number of time steps, 
    Pillar indicates whether the central pillar is displayed, note this won't change whether the pillar is present (to do that set κsteric=0)
    V0mLeftDiff0 indicates the volume of cytoplasm initialy confined in the left chamber (preferred volume)
    κmemL0 is the initial contractility of the rear membrane wall
    Mirror indicates whether to reflect the plotted output to show the whole cell, rather than just the top half
    Loud toggles how much is outputted, set true for debugging
    RhoBurst specifies whether the rhoburst should be initially triggered
    Params is the constructor containing cell information
    σML0 is the initial membrane contractility on the left
    KeepGoing overrides the code which ends the simulation after transmigration 

"""
function main(x0vec::Array; T=30::Int, Pillar=true::Bool, V0cytLeft=V0cytLeft::Float64, σSL=0.0::Float64, Mirror=false::Bool, Loud=false::Bool, RhoBurst=false::Bool, Params=Params::CellParameters, σML0=Params.σML0, KeepGoing=false::Bool, timer=1.0::Float64, Clamping=1000.::Float64)
    t0 = 0.0
    timer = 1.0 
    Clamping = 1000.
    Pullback = true
    @unpack σMT, V0cyt, α, β, ct, R, V0n, Δt, BurstAddition, ρN, ρmem, σN, ηC, ηCM, VMT, η, κsteric, κintersect, ξ, μ, σS0, ForceM, ηM, ξmem, Maxmyosin, W, PCutoff, τ, βbar, VCollapse, σMBG, Force, NewPressure, MTdecay = Params
    RunVars = [(V0cytLeft=V0cytLeft, σML=σML0, RhoBurst=RhoBurst, σSL=σSL)]
    plist = []
    Loud && println("V0 is $(round(V0n, digits=3))")
    PosList = Vector{Float64}[x0vec]


    X0 = x0vec[1:N]
    Y0 = x0vec[N+1:2N]
    X0m = x0vec[2N+1:2N+M]
    Y0m = x0vec[2N+M+1:end]
    COM0 = getCOM(X0, Y0)
    p0 = MakePlot(X0, Y0, X0m, Y0m, Pillar=Pillar, ct1=ct[1], ct2=ct[2], R=R, Mirror=Mirror, σML=σML0, σMT=σMT, Close=true, ContGradient=true, Maxmyosin=Maxmyosin)
    plot!(title = " t = $(fillzeros(string(t0), 2))")
    annotate!(relative(p0, 0.2, 0.95)..., text("Occluded: $(Occluded(COM0))"))
    annotate!(relative(p0, 0.2, 0.85)..., text("RhoBurst: $RhoBurst"))
    annotate!(minimum(X0m), -0.4, text("$(round(σML0+σSL, digits=2))", 10), :left)
    annotate!(maximum(X0m), -0.4, text("$(round(σMT/2, digits=2))", 10), :right)
    annotate!(relative(p0, 0.2, 0.75)..., text("σN: $σN"))
    annotate!(relative(p0, 0.2, 0.65)..., text("β: $β"))
    display(p0)
    push!(plist, p0)
    VCushion = VCollapse + (VMT - VCollapse)*timer
    for t in 1:T
        time0 = time()
        println("### time $t of $T")
        X0 = x0vec[1:N]
        Y0 = x0vec[N+1:2N]
        X0m = x0vec[2N+1:2N+M]
        Y0m = x0vec[2N+M+1:end]

        X0m, Y0m = Rebalance(X0m, Y0m)
        
        V0cytLeft = RunVars[t].V0cytLeft
        σML = RunVars[t].σML
        σSL = RunVars[t].σSL

        @show σSL
        xvec = [X0; Y0[2:end-1]; X0m; Y0m[2:end-1]]

        result = optimize(
                x -> EnergyTotal(x, x0vec, Params=Params, V0cytLeft=V0cytLeft, σML=σML, σSL=σSL, RhoBurst=RhoBurst, Pullback=Pullback, Clamping=Clamping),
                xvec, BFGS(linesearch=LineSearches.HagerZhang()), autodiff=:forward,
                Optim.Options(iterations=Int(1e8), g_tol=1e-16, allow_f_increases=true, time_limit=300)
            )
        
        time1 = time()
        Loud && println("Solving the optimisation took $(time1-time0) seconds")
        println("Optimisation took $(round(Optim.iterations(result), digits=3)) iterations")

        println("Cost is $(Optim.minimum(result))")
        xvec = Optim.minimizer(result)

        Loud && EnergyTotal(xvec, x0vec, V0cytLeft=V0cytLeft, σML=RunVars[t].σML, σSL=RunVars[t].σSL, Loud=true, Clamping=Clamping)

        X = xvec[1:N]
        Y = xvec[N+1:2N-2]
        Y = [0.0; Y; 0.0]
        Xm = xvec[2N-1:2N+M-2]
        Ym = xvec[2N+M-1:end]
        Ym = [0.0; Ym; 0.0]

        V = TrapezoidalQuadrature(X, Y)
        Loud && println("Nucleus Volume is $(round(V, digits=3))")
        COM = getCOM(X, Y)
        COMold = getCOM(X0, Y0)
        Loud && println("Nucleus Centre of Mass is $(round(COM, digits=3))")

        Vm = TrapezoidalQuadrature(Xm, Ym)
        Loud && println("Cell Volume is $(round(Vm, digits=3))")
        COMm = getCOM(Xm, Ym)
        Loud && println("Cell Centre of Mass is $(round(COMm, digits=3))")

        Occ, nclose = Occluded(X, Y)
        Occ0, nclose = Occluded(X0, Y0)
        VL, VR = AreaLeftRight(X, Y, x0=X[nclose])
        VLm, VRm = AreaLeftRight(Xm, Ym, x0=X[nclose])

        mclose = findfirst(m->X0m[m] > X0[nclose], 1:M)

        if (Loud && Occ0)
            println("Preferred cytoplasm volume left is $(V0cytLeft), after optimisation current value is $(VLm-VL)")
            println("Preferred cytoplasm volume right is $(V0cyt-V0cytLeft), after optimisation current value is $(VRm-VR)")
        end
        if Occ0 && !Occ
            Pullback=false
        end
        
        Loud && println("Nuclear COM moved a distance of $(COM-COMold)") 

        Pleft = ρmem * (V0cytLeft - (VLm - VL))
        if NewPressure
            Pright = ρmem * (V0cyt - (VLm - VL) - (VRm - VR))
        else
            Pright = ρmem * (V0cyt - V0cytLeft - (VRm - VR))
        end
        Pmic = τ * max(0.0, (VMT - (VLm - VL)))
        ΔP = Pright - Pleft 
        @show ΔP

        if (Occ0 && Pleft + Pmic > PCutoff)
            RhoBurst = true
            println("Now switching on RhoBurst")
        end
        if Occ && Occ0
            NewVolL = V0cytLeft + α*ΔP*Δt
            !RhoBurst && (NewVolL = max(NewVolL, VMT))
            NewσML = RunVars[t].σML + (β*(σMT - RunVars[t].σML) + βbar*(σMT-2RunVars[t].σML)) * Δt
        else
            NewVolL = VLm-VL
            NewσML = RunVars[t].σML+β*(0.5*σMT-RunVars[t].σML)*Δt
        end
        
        if RhoBurst
            timer = timer - MTdecay*timer*Δt

            NewVolL = max(NewVolL, VCushion)

            NewσSL = BurstAddition*(σML+σMBG)*(1-timer)
            VCushion = VCollapse + (VMT - VCollapse)*timer
        else
            NewσSL = RunVars[t].σSL-β*RunVars[t].σSL*Δt
        end
        Loud && println("Saving the left Volume $NewVolL") 
        push!(RunVars, (V0cytLeft=NewVolL, σML=NewσML, RhoBurst=RhoBurst, σSL=NewσSL))

        p = MakePlot(X, Y, Xm, Ym, Pillar=Pillar, ct1=ct[1], ct2=ct[2], R=R, Mirror=Mirror, σML=RunVars[end].σML, σMT=σMT, Close=true, ContGradient=true, Maxmyosin=Maxmyosin)
        plot!(title = " t = "*fillzeros(string(round(t0 + t*Δt, digits=2)), 2))

        scatter!([Xm[mclose]], [Ym[mclose]], markershape=:star, color=:black, label=:none, markersize=10)

        annotate!(relative(p, 0.2, 0.95)..., text("Occluded: $Occ"))
        annotate!(relative(p, 0.2, 0.85)..., text("RhoBurst: $(RunVars[t].RhoBurst)"))
        annotate!(minimum(Xm), -0.4, text("$(round(RunVars[t].σML+RunVars[t].σSL, digits=2))", 10), :right)
        annotate!(maximum(Xm), -0.4, text("$(round(σMT-RunVars[t].σML, digits=2))", 10), :left)
        annotate!(relative(p, 0.2, 0.75)..., text("σN: $σN"))
        annotate!(relative(p, 0.2, 0.65)..., text("PRear: $(round(Pleft, digits=2))"))
        annotate!(relative(p, 0.2, 0.55)..., text("PFront: $(round(Pright, digits=2))"))

        display(p)
        push!(plist, p)
        x0vec = [X; Y; Xm; Ym]
        push!(CurrentRun, x0vec)
        push!(PosList, x0vec)
        println("Step took $(round(time() - time0, digits=3)) seconds \n")
        if !KeepGoing && X[1] > ct[1]+R
            break
        end
    end

    push!(Runs, PosList)
    push!(NMs, (N,M))
    push!(Vars, RunVars)
    push!(Consts, Params)
    return 
end


function RunSim(; T=200::Int, Clamping=1000., args...)
    Params = CellParameters(V0n=V0n, V0cyt=V0cyt; args...)
    main(x0vec, T=T, Params=Params, Clamping=Clamping)
end
