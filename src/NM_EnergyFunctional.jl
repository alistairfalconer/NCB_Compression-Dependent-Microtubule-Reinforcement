"""
    Energy functional for the nucleus and membrane update. Accepts as arguments xvec containing the positions of nucleus and membrane, and x0vec containing the previous nucleus and membrane position. Optional arguments: 
        Loud determines if output is verbose 
        V0mLeftDiff is the amount of cytoplasm confined in the left chamber, configuring the preferred volume of that chamber if the cell is confined
        σML is the current contractility on the left half of the membrane
"""
function EnergyTotal(xvec::Array, x0vec::Vector{Float64}; Loud=false::Bool, V0cytLeft=V0cytLeft::Float64, σML=σML::Float64, σSL=0.0::Float64, RhoBurst=false::Bool, Params=Params::CellParameters, Pullback=true::Bool, Clamping=1000.)

    @unpack σMT, V0cyt, α, β, ct, R, V0n, Δt, BurstAddition, ρN, ρmem, σN, ηC, ηCM, VMT, η, κsteric, κintersect, ξ, μ, σS0, ForceM, ηM, ξmem, Maxmyosin, W, PCutoff, τ, σMBG, Force, NewPressure, ChangeForce = Params


    Loud && println("σSL is $σSL")

    if any(isnan.(xvec))
        return Inf
    end
    X, X0 = xvec[1:N], x0vec[1:N]
    Y, Y0 = xvec[N+1:2N-2], x0vec[N+1:2N]
    Y = [0.0; Y; 0.0]
    Xm, X0m = xvec[2N-1:2N+M-2], x0vec[2N+1:2N+M]
    Ym, Y0m = xvec[2N+M-1:end], x0vec[2N+M+1:end]
    Ym = [0.0; Ym; 0.0]

    #centres of mass
    COM = getCOM(X, Y)
    COMold = getCOM(X0, Y0)
    COMm = getCOM(Xm, Ym)
    COMoldm = getCOM(X0m, Y0m) 

    Occ = Occluded(COM)
    Occ0, nclose = Occluded(X0, Y0)
    mclose = findfirst(m->X0m[m] > X0[nclose], 1:M)

    #Preserve nucleus volume
    V = TrapezoidalQuadrature(X, Y)
    EV = ((ρN * (V-V0n)^2))#::Float64

    VLeft, VRight = AreaLeftRight(X, Y, x0=X[nclose])
    VmLeft, VmRight = AreaLeftRight(Xm, Ym, x0=X[nclose])

    VLeftcyt, VRightcyt = VmLeft-VLeft, VmRight-VRight

    EVmTotal = (ρmem * (VLeftcyt + VRightcyt - V0cyt)^2)
    
    if Occ0
        EVmLeft = (ρmem * (VLeftcyt - V0cytLeft)^2)
        if NewPressure
            EVmRight = (ρmem * (VRightcyt - (V0cyt - VLeftcyt))^2)
        else
            EVmRight = (ρmem * (VRightcyt - (V0cyt - V0cytLeft))^2)
        end
        EVmTotal = EVmLeft + EVmRight
    else
        
    end

    Etest = 0.0

    IntervalLengths = norm.(zip(diff(X), diff(Y)))
    EE = ξ*sum((IntervalLengths .- mean(IntervalLengths)).^2)/N

    IntervalLengthsm = norm.(zip(diff(Xm), diff(Ym)))
    EEm = 0.

    IntervalLengthsmLeft = IntervalLengthsm[1:mclose-2]
    IntervalLengthsmRight = IntervalLengthsm[mclose-1:end]

    if Occ0
        EEm += (1+BurstAddition*RhoBurst)*ξmem*sum((IntervalLengthsmLeft .- mean(IntervalLengthsmLeft)).^2)/length(IntervalLengthsmLeft)+ 10*(1+BurstAddition*RhoBurst)*ξmem*sum((IntervalLengthsmRight .- mean(IntervalLengthsmRight)).^2)/length(IntervalLengthsmRight)
    else
        EEm += ξmem*sum((IntervalLengthsm .- mean(IntervalLengthsm)).^2)/M
    end

    #minimize perimeter
    L = sum(IntervalLengths)
    EL = (σN*L) 

    Ll = sum(IntervalLengthsmLeft) 
    Lr = sum(IntervalLengthsmRight)
    ELm = (σML + σSL + σMBG) * Ll + (σMT - σML + σMBG) * Lr

    #forcing term
    if ChangeForce
        EF = Occ0 ? -Force * (COM) : 0.0
        EFm = Occ0 ? -(ForceM-Force) * (COMm) : -ForceM * COMm
    else
        EF =  -Force * (COM)
        EFm = -(ForceM-Force) * (COMm) 
    end

  
  
    #friction term
    EFric = η*((COM - COMold) - (COMm - COMoldm))^2/2/Δt
    Efricm = ηM*(COMm - COMoldm)^2/2/Δt
    

    #Cage energy imposes minimum on VLeft
    ECage = RhoBurst ? τ*max(0., (VMT/2 - VLeftcyt))^2 : τ*max(0., (VMT - VLeftcyt))^2 

    #steric energy
    ESteric = 0.0
    EStericm = 0.0
    EIntersection = 0.0
    
    fixnode1, fixnode1_0 = [Xm[mclose-1], Ym[mclose-1]], [X0m[mclose-1], Y0m[mclose-1]]
    fixnode2, fixnode2_0 = [Xm[mclose], Ym[mclose]], [X0m[mclose], Y0m[mclose]]

    if Occ0
        ESteric += Clamping * (1+RhoBurst*BurstAddition) *norm(fixnode1_0 .- fixnode1)
        ESteric += Clamping * (1+RhoBurst*BurstAddition) *norm(fixnode2_0 .- fixnode2)
    end

    
    for m in 1:M
        m in 5:M-4 && (EStericm += κsteric*min(0.0, Ym[m]-1.25)^2)
        (Loud && m in [4:M-3]) && println("m is $m, Y is $(Ym[m])")
        EStericm += κsteric*max(0.0, Ym[m]-1.5ct[2])^2
        if norm(ct .- [Xm[m], Ym[m]]) < R-W
            EStericm += κsteric*(R-W-norm(ct .- [Xm[m], Ym[m]]))^2
        end
    end
    for n in 1:N
        ESteric += (1000*min(0.0, Y[n])^2)
        if norm(ct .- [X[n], Y[n]]) < R
            ESteric += κsteric*(R-norm(ct .- [X[n], Y[n]]))^2
        end
        EIntersection += (κintersect*max(0.0, Xm[1]-X[n])^2)
        EIntersection += (κintersect*max(0.0, X[n]-Xm[end])^2)
        for m in 1:M-1
            if (X0m[m+1] - X0[n])*(X0[n] - X0m[m]) >= 0
                x1, x2, y1, y2 = Xm[m], Xm[m+1], Ym[m], Ym[m+1]
                yx = (y2-y1)/(x2-x1)*(X[n]-x1) + y1
                EIntersection += (κintersect*max(0.0, Y[n] - yx)^2)
            end
        end
    end

    ECurve, Ecurvem = 0., 0.
    if ηC > 0 || ηCM > 0
        println("Got Here")
        #Preserve curvature
        Cvals = Curvature(X, Y)
        Cvals0 = Curvature(X0, Y0)
        ECurve = (ηC * (sum((Cvals .- Cvals0).^2))/2/Δt)

        Cvalsm = Curvature(Xm, Ym)
        Cvals0m = Curvature(X0m, Y0m)
        Ecurvem = (ηCM * (sum(abs.(Cvalsm .- Cvals0m).^2))/2/Δt)
    end

    if Pullback
        ECentre = (μ*(COM - COMm)^2)
    else
        ECentre = (μ*min(0.0, (COM - COMm))^2)
    end


    if Loud
        println("Nucleus volume energy is $EV")
        println("Total volume energy is $EVmTotal")
        println("Nucleus perimeter energy $EL")
        println("Membrane perimeter energy is $ELm")
        println("Nucleus forcing energy is $EF")
        println("Membrane Forcing energy is $EFm")
        println("Nucleus extension energy is $EE")
        println("Membrane extension energy is $EEm")
        println("Nucleus friction energy is $EFric")
        println("Membrane Friction energy is $Efricm")
        println("Nucleus curvature energy is $ECurve")
        println("Membrane Curvature energy is $Ecurvem")
        println("Nucleus centring energy is $ECentre")
        println("Nucleus steric energy is $ESteric")
        println("Membrane steric energy is $EStericm")
        println("Total intersection energy is $EIntersection")   
        println("Centring energy is $ECentre")  
        println("Caging energy is $ECage") 
        println("Debugging energy is $Etest")   
    end
    Eng = EV + EVmTotal + EL + EF + EE + EFric + ESteric + EIntersection + ECurve + ECentre + ELm + EEm + Efricm + Ecurvem + EFm + EStericm + Etest + ECage
    Loud && println("Total energy is: $Eng")
    return Eng
end

EnergyTotal(X::Vector{Float64}, Y::Vector{Float64}, Xm::Vector{Float64}, Ym::Vector{Float64}, x0vec::Vector{Float64}; args...) = EnergyTotal([X;Y[2:end-1];Xm;Ym[2:end-1]], x0vec; args...)