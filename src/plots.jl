module HybridSpikeSorterPlots
using HybridSpikeSorter
using HMMSpikeSorter
using PyPlot
using GLAbstraction
using GLVisualize
using GLWindow
using GeometryTypes
using Colors
using Reactive

function plot_data(fdata::Signal{Vector{Float64}}, y::Signal{Matrix{Float64}}, cids::Signal{Vector{Int64}}, waveforms::Signal{Matrix{Float64}})
    fig = plt[:figure]()
    ax1 = fig[:add_subplot](221, projection="3d")
    ax2 = fig[:add_subplot](222)
    ax3 = fig[:add_subplot](223)
    map(fdata) do _fdata
        if !isempty(_fdata)
            ax3[:clear]()
            ax3[:plot](fdata)
        end
    end
    map(waveforms) do _waveforms
        if !isempty(_waveforms)
            ax2[:clear]()
            ax2[:plot](_waveforms;color="blue")
        end
    end
    map(y,cids) do _y, _cids
        if !isempty(_y)
            ax1[:clear]()
            if isempty(_cids)
                ax1[:scatter](_y[1,:], _y[2,:], _y[3,:];s=1.0)
            else
                clusterids = unique(cids)
                sort!(clusterids)
                nclusters = maximum(clusterids)
                 _colors = [(cc.r, cc.g, cc.b) for cc in distinguishable_colors(nclusters, colorant"tomato";lchoices=linspace(10,80,15))]
                 ax1[:scatter](_y[1,:], _y[2,:], _y[3,:];s=1.0, c=_colors[cids])
             end
         end
     end
end

function plot_clusters(Y::Signal{Matrix{Float64}}, cids::Signal{Vector{Int64}})
    window = glscreen("Cluster Pot", resolution=(1024,1024))
    _colors = map(Y, cids) do _Y, _cids
        if isempty(_cids)
            cc = [Point3f0(0.4f0, 0.3f0, 0.1f0) for i in 1:size(_Y,2)]
        else
            clusterids = unique(_cids)
            sort!(clusterids)
            nclusters = maximum(clusterids)
             cluster_colors = distinguishable_colors(nclusters, colorant"tomato";lchoices=linspace(10,80,15))
             cc = Array{Point3f0}(length(_cids))
             for i in 1:length(cc)
                 q = cluster_colors[_cids[i]]
                 cc[i] = Point3f0(Float32(q.r), Float32(q.g), Float32(q.b))
             end
        end
        cc
    end
    _points = map(Y) do _Y
        [Point3f0(Float32(_Y[1,i]), Float32(_Y[2,i]), Float32(_Y[3,i])) for i in 1:size(_Y,2)]
    end
    vpoints = visualize((Circle, _points), color=_colors)
    const robj = vpoints.children[]
    const gpu_colors = robj[:color]
    const m2id = GLWindow.mouse2id(window)
    @materialize mouseposition, mouse_buttons_pressed = window.inputs
    preserve(map(mouse_buttons_pressed) do aa
                 id, index = value(m2id)
                 if id == robj.id && 0 < index < length(gpu_colors)
                     _cid = value(cids)[index]
                     gpu_colors[index] = Point3f0(0.95f0, 0.1f0, 0.45f0)
                 end
                 return index
             end)

    _view(vpoints, window,camera=:perspective)
    @async GLWindow.waiting_renderloop(window)
    nothing
end

function plot(waveforms::Matrix{Float64}, cids::Vector{Int64}, modelf::HMMSpikeSorter.HMMSpikingModel)
    clusterids = unique(cids)
    sort!(clusterids)
    nclusters = maximum(clusterids)
    #plot in PCA space
    pca = fit(PCA, waveforms)
    y = transforms(pca, waveforms)
    fig = plt[:figure]()
    ax1 = fig[:add_subplot](2,2,1,projection="3d")
     _colors = [(cc.r, cc.g, cc.b) for cc in distinguishable_colors(nclusters, colorant"tomato";lchoices=linspace(10,80,15))]
     ax1[:scatter](y[1,:], y[2,:], y[3,:];s=1.0,c=_colors[cids])
     ll = DirichletProcessMixtures.lratio(cids, y)
     ax2 = fig[:add_subplot](2,2,2)
     ax2[:bar](clusterids, [ll[k] for k in clusterids];color=_colors)

     ax3 = fig[:add_subplot](2,2,3)
     for (i,c) in enumerate(clusterids)
         _idx = cids .== c
         μ = mean(y[:,_idx],2)[:]
         ax3[:plot](μ;color=_colors[c],label="Cluster $c")
     end
     ax3[:legend]()
     
     ax4 = fig[:add_subplot](224)
     ax4[:plot](model_response(modelf)[1:80_000])
     ax4[:plot](predict(modelf)[1:80_000])
     fig
end

end#module

