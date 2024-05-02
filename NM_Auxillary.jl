"""
    numerically integrate points yvals assuming even spacing of h in the x direction
"""
function SimpsonsQuadrature(yvals::Vector{Float64}, h::Float64)::Float64
    N=length(yvals)::Int
    return h/3 * (yvals[1] + 2sum(yvals[Int(2j+1)] for j in 1:(N/2-1)) + 4sum(yvals[Int(2j)] for j in 1:(N-1)/2) + yvals[N])
end

"""
    find area under curve f within a <= x <= b using Simpsons method
    Optional argument N determines number of nodes to use in the calculation
"""
function SimpsonsQuadrature(f::Function, a::Float64, b::Float64; N=10001::Int)
    xvals = range(a, stop=b, length=N)
    h = xvals[2] - xvals[1]
    return SimpsonsQuadrature(f.(xvals), h)
    #return h/3 * (f(xvals[1]) + 2sum(f(xvals[Int(2j+1)]) for j in 1:(N/2-1)) + 4sum(f(xvals[Int(2j)]) for j in 1:(N-1)/2) + f(xvals[N]))
end

"""
    Find area under curve using Trapezoidal method, use this rather than Simpsons method when given points that aren't evenly spaced in x direction
"""
function TrapezoidalQuadrature(xvals::Vector, yvals::Vector)
    if length(xvals) != length(yvals)
        println("############## Trapezoidal Quad: lengths mismatch")
        return
    end
    if length(xvals) == 1
        return 0
    end
    n = length(xvals)-1
    return sum((yvals[k] + yvals[k+1])*(xvals[k+1] - xvals[k]) for k in 1:n)/2
end

"""
    Calculate the curvature at each node of the curve specified by arrays xvals, yvals, returns an array of same length as xvals, yvals
""" 
function Curvature(xvals::Vector, yvals::Vector)::Vector
    if length(xvals) != length(yvals)
        println("### Curvature: Dimension Mismatch")
        return 
    end

    Curvature = zeros(length(xvals)-1)
    Curvature[1] = -(xvals[2]-2xvals[1]+xvals[2])*(yvals[2]+yvals[2])/2 + (yvals[2]-2yvals[1]-yvals[2])*(xvals[2]-xvals[2])/2

    for n in 2:length(xvals)-1
        Curvature[n] = -(xvals[n+1]-2xvals[n]+xvals[n-1])*(yvals[n+1]-yvals[n-1])/2 + (yvals[n+1]-2yvals[n]+yvals[n-1])*(xvals[n+1]-xvals[n-1])/2
    end
    return Curvature
end

"""
    locally average the vector x
"""
av(x::Vector)::Vector = (x + circshift(x, -1))/2
av2(x::Vector)::Vector = (x[1:end-1] + x[2:end])/2
"""
    get the COM of the shape specified by nodes x, y
"""
function getCOM(x::Vector, y::Vector)
    xp=diff([x; x[1]])
    yp=diff([y; y[1]])
    lens=sqrt.(xp.^2+yp.^2)
    #println(minimum(lens))
    nx= -yp ./ sqrt.(xp.^2 .+ yp.^2)
    ny= xp ./ sqrt.(xp.^2 + yp.^2)
    Mx=sum(nx.*av(x).^2/2 .*lens)
    area=sum((nx.*av(x)/2 .+ny.*av(y)/2).*lens)
    return Mx/area
end

"""
    Get the COM of the shape specified by [x; y]
"""
function getCOM(xvec::Vector{Float64})::Float64
    local N = Int(length(xvec)/2)
    return getCOM(xvec[1:N], xvec[N+1:2N])
end

"""
    given a shape specified by the nodes xvals, yvals, find the area on either side of the point x0. There is some effort to handle cases where the first point is to the right of the boundary but the second is to the left etc.
    Optional argument x0 specifies the cutoff point
"""
function AreaLeftRight(xvals::Vector, yvals::Vector; x0=0.0::Float64)#::Tuple{Float64, Float64}
    local N = length(xvals)
    @assert length(yvals) == N "Dimension mismatch in AreaLeftRight"
    if maximum(xvals) <= x0
        return TrapezoidalQuadrature(xvals, yvals), 0.0
    elseif minimum(xvals) >= x0
        return 0.0, TrapezoidalQuadrature(xvals, yvals)
    end
    nbreak = 0
    if xvals[1] >= x0
        return 0., TrapezoidalQuadrature(xvals, yvals)
    end
    for n in 2:N
        if abs(xvals[n]-x0) < 1e-3
            return TrapezoidalQuadrature(xvals[1:n], yvals[1:n]), TrapezoidalQuadrature(xvals[n:end], yvals[n:end])
        end
        if xvals[n] > x0
            nbreak = n
            break
        end
    end
    x1, y1, x2, y2 = xvals[nbreak-1], yvals[nbreak-1], xvals[nbreak], yvals[nbreak]
    y0 = (x0-x1)*(y2-y1)/(x2-x1)+y1
    if nbreak == 2
        return (x0-x1)*(y1+y0)/2, TrapezoidalQuadrature(xvals[nbreak:end], yvals[nbreak:end]) + x2*(y0+y2)/2
    end
    AL = TrapezoidalQuadrature(xvals[1:max(nbreak-1, 1)], yvals[1:max(nbreak-1, 1)]) + (x0-x1)*(y1+y0)/2
    AR = TrapezoidalQuadrature(xvals[nbreak:end], yvals[nbreak:end]) + (x2-x0)*(y0+y2)/2
    return AL, AR
end


"""
    same as sum but will return 0.0 rather than an error if the input array is empty
"""
function mysum(A::Vector)
    if length(A) == 0
        return 0.0
    end
    return sum(A)
end

"""
    For the curve specified by nodes xvals, yvals, calculate the length on either side of the cutoff point x0. There is some effort to handle cases where the first point is to the right of the boundary but the second is to the left etc.
    Optional argument x0 specifies the cutoff point 
"""
function LengthLeftRight(xvals::Array, yvals::Array; x0=0.0::Float64)
    local N = length(xvals)
    if length(yvals) != N
        println("#### Dimension mismatch in AreaLeftRight")
    end
    nbreak = 0
    diffx, diffy = diff(xvals), diff(yvals)
    if maximum(xvals) <= x0
        return sum(norm([diffx[i], diffy[i]]) for i in 1:N-1), 0.0
    elseif minimum(xvals) >= x0
        return 0.0, sum(norm([diffx[i], diffy[i]]) for i in 1:N-1)
    end
    for n in 1:N
        if abs(xvals[n]-x0) < 1e-3
            return mysum(collect(norm([diffx[i], diffy[i]]) for i in 1:n-1)), mysum(collect(norm([diffx[i], diffy[i]]) for i in n:N-1))
        end
    end
    Right = [true; diff(xvals) .> 0]
    if !any((xvals .< x0) .* Right)
        return 0.0,sum(norm([diffx[i], diffy[i]]) for i in 1:N-1)
    end
    if !any((xvals .> x0) .* circshift(Right, -1))
        return sum(norm([diffx[i], diffy[i]]) for i in 1:N-1), 0.0
    end
    nbreak = findlast((xvals .< x0) .* Right)+1
    x1, y1, x2, y2 = xvals[nbreak-1], yvals[nbreak-1], xvals[nbreak], yvals[nbreak]
    y0 = (x0-x1)*(y2-y1)/(x2-x1)+y1
    LL = mysum(collect(norm([diffx[i], diffy[i]]) for i in 1:nbreak-2)) + norm([x0-x1, y0-y1])
    LR = mysum(collect(norm([diffx[i], diffy[i]]) for i in nbreak:N-1)) + norm([x2-x0, y2-y0])
    return LL, LR
end

function getRl(RunIndex::Int)
    RunIndex = RImod(RunIndex)
    xvec = Runs[RunIndex][1]
    N, M = NMs[RunIndex]
    return (xvec[N] - xvec[1])/2
end

function Occluded(COM; ct=ct::Array{Float64, 1}, R=R::Float64)
    @assert length(X) == length(Y) "Occluded test given vectors of unequal size"
    return norm([ct[1]-COM, ct[2]]) < 1.05(R + Rl)
end

function Occluded(X::Vector, Y::Vector; ct=ct::Array{Float64, 1}, R=R::Float64)
    @assert length(X) == length(Y) "Occluded test given vectors of unequal size"
    COM = getCOM(X, Y)
    Nodes = [(X[i], Y[i]) for i in 1:length(X)] 
    return Occluded(COM, ct=ct, R=R), argmin(norm.([Node .- ct for Node in Nodes]))
end

function Occluded(RI::Int, t::Int)
    RI = RImod(RI)
    COM = getCOM(Runs[RI][t])
    return Occluded(COM, ct=Consts[RI].ct, R=Consts[RI].R)
end

hillfn(x; n=10, θ=0.3) = x^n/(x^n + θ^n)
function HillContractility(σL::Float64, σR::Float64; M=M)
    svals = range(1.0, stop=0.0, length=M)
    hillvals = (x->hillfn(x)).(svals)
    return (1 .- hillvals)*(σL-σR) .+ σR
end


function FindNodeStiffness(σML, σMT; M=M, nclose=N/2)
    NodeStiffness = range(σML, stop=(σMT-σML), length=M)
    return NodeStiffness
end

function Rebalance(Xm::Vector{Float64}, Ym::Vector{Float64})
    IntervalLengths = norm.(zip(diff(Xm), diff(Ym)))
    mi, Mi = argmin(IntervalLengths), argmax(IntervalLengths)
    if IntervalLengths[mi] < IntervalLengths[Mi]/3
        println("moving $mi -> $Mi")
        newpos = [Xm[Mi]+Xm[Mi+1], Ym[Mi]+Ym[Mi+1]]/2
        return [Xm[1:mi]; Xm[mi+2:Mi]; newpos[1]; Xm[Mi+1:end]], [Ym[1:mi]; Ym[mi+2:Mi]; newpos[2]; Ym[Mi+1:end]]
    end
    return Xm, Ym
end

function FullRebalance(Xm::Vector{Float64}, Ym::Vector{Float64})
    Xmc, Ymc = Vector(Xm), Vector(Ym)
    for _ in 1:10
        Xm2, Ym2 = Rebalance(Xmc, Ymc)
        all((Xmc, Ymc) .≈ (Xm2, Ym2)) && return Xm2, Ym2
        Xmc, Ymc = Xm2, Ym2
    end
    return Xmc, Ymc
end 


#find length of continuous function f between 2 points, specified by x values x1, x2. This is approximated by discretising the curve into M nodes and approximating as a straight line between them
function ArcLength(f::Function, x1::Float64, x2::Float64; M=1000::Int)
    if x1 == x2
        return 0.0
    end
    X = range(min(x1, x2), stop=max(x1, x2), length=M)
    Y = f.(X)
    return sum(norm([X[m+1],Y[m+1]] .- [X[m],Y[m]]) for m in 1:M-1)
end


#return array of the lengths of the path along function f with nodes specified by the x values in X, using the same approximation as in ArcLength. This can be used to check the output of EvenSpacing
function SubLengths(f::Function, X::Array)
    return [ArcLength(f, X[n], X[n+1]) for n in 1:N-1]
end

#return a list of N x positions with a=x1<x2<...<xN=b, such that the nodes (xn, f(xn)) are evenly spaced along the function x
function EvenSpacing(a::Float64, b::Float64, f::Function, N::Int)
    L = ArcLength(f, a, b, M=Int(1e6))
    ℓ = L/(N-1)#ideal length between each node
    X = zeros(N)
    X[1], X[end] = a, b
    for n in 2:N-1
        result = optimize(
            x -> (ArcLength(f, X[n-1], x) - ℓ)^2, #objective function (to be minimised)
            X[n-1], #left boundary on solution
            b #right boundary on solution
        )
        X[n] = Optim.minimizer(result)#extract minimising value
    end
    return X
end


