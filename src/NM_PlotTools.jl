"""
    For help with annotation, accepts a plot and returns the point which has relative position (rx, ry), where rx, ry are numbers between 0 and 1. 
"""
function relative(p::Plots.Plot, rx::Float64, ry::Float64)
    xlims = Plots.xlims(p)
    ylims = Plots.ylims(p)
    return xlims[1] + rx * (xlims[2]-xlims[1]), ylims[1] + ry * (ylims[2] - ylims[1])
end

"""
    Accept a string of a decimal number and append the needed number of zeros such that the there are num digits after the decimal place
"""
fillzeros(in::String, num::Int)::String = length(split(in, ".")[2])>=num ? split(in, ".")[1]*"."*split(in, ".")[2][1:num] : in*"0"^(num-length(split(in,".")[2]))


"""
    create a standard plot of the membrane
"""
function MakePlot(X::Vector, Y::Vector, Xm::Vector, Ym::Vector; Pillar=true::Bool, ct1=ct[1]::Float64, ct2=ct[2]::Float64, R=R::Float64, Mirror=true::Bool, σML=(-1.0)::Float64, σMT=(-1.0)::Float64, Close=true::Bool, ContGradient=true::Bool, Maxmyosin=0.9::Float64)
    p = plot(aspect_ratio=1, legend=:none)
    if Pillar
        θvals = range(0, stop=2π, length=1000)
        plot!(ct1 .+ (R-W)*cos.(θvals), ct2 .+ (R-W)*sin.(θvals), label=:none, color=:black, linewidth=3)
        Mirror && plot!(ct1 .+ (R-W)*cos.(θvals), -ct2 .- (R-W)*sin.(θvals), label=:none, color=:black, linewidth=3)
        plot!(ct1 .+ (R)*cos.(θvals), ct2 .+ (R)*sin.(θvals), label=:none, color=:black, linewidth=1)
        Mirror && plot!(ct1 .+ (R)*cos.(θvals), -ct2 .- (R)*sin.(θvals), label=:none, color=:black, linewidth=1)
    end
    plot!(X, Y, label="Nucleus", color=1, markershape=:circle)
    plot!(Xm, Ym, label="Membrane", color=2, markershape=:circle)
    COM = getCOM(X, Y)
    scatter!([COM], [0.0], marker=:star, label=:none, color=1)
    COMm = getCOM(Xm, Ym)
    scatter!([COMm], [0.0], marker=:star, label=:none, color=2)
    if Mirror
        plot!(X, -Y, label=:none, color=1, markershape=:circle)
        plot!(Xm, -Ym, label=:none, color=2, markershape=:circle)
        
    end
    Occ, nclose = Occluded(X, Y, ct=[ct[1], ct[2]], R=R)
    if σMT > 0
        colors = cgrad([:green, :orange])
        NodeStiffness = FindNodeStiffness(σML, σMT, M=length(Xm))
        σmin, σmax = (1-Maxmyosin)σMT, Maxmyosin*σMT
        colArray = [colors[NS/(σmax-σmin) - σmin/(σmin+σmax)] for NS in NodeStiffness]
        scatter!(Xm, Ym, color=colArray, label=:none)
        Mirror && scatter!(Xm, -Ym, color=colArray, label=:none)

    end
    Close && scatter!([X[nclose]], [Y[nclose]], color=1, markershape=:star, markersize=10, label=:none)
    (Mirror && Close) && scatter!([X[nclose]], [-Y[nclose]], color=1, markershape=:star, markersize=10, label=:none)
    return p
end 

"""
    Returns a tuple containing (X, Y, Xm, Ym) for the specified timestep of the run indexed by RunIndex
"""
function GetXY(RunIndex::Int, t::Int)::Tuple
    if RunIndex < 1
        RunIndex = length(Runs) + RunIndex
    end
    local N, M = NMs[RunIndex]
    xvec = Runs[RunIndex][t+1]
    return xvec[1:N], xvec[N+1:2N], xvec[2N+1:2N+M], xvec[2N+M+1:end]
end

GetXY(t::Int)::Tuple = GetXY(length(Runs), t)

function myscatter(xdata, ydata; kwargs...)
    scatter(xdata, ydata; kwargs...)
    ynan = findall(isnan, ydata)
    !isempty(ynan) && vline!(xdata[ynan], ls=:dash, lc=:black)
    return plot!()
end



