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
using StatsBase

function plot_sorting(feature_model, feature_data::Matrix{Float64}, spike_model,clusterid::Array{Int64,1};max_lratio=20.0)
    fig = plt[:figure]()
    ax1 = fig[:add_subplot](221, projection="3d")
    ax2 = fig[:add_subplot](222)
    ax3 = fig[:add_subplot](223)
    ax4 = fig[:add_subplot](224)
    ax3[:clear]()
    fdata = model_response(spike_model)
    S = predict(spike_model)
    ax3[:plot](fdata[1:100_000])
    ax3[:plot](S[1:100_000])

    ax1[:clear]()
    cids = HybridSpikeSorter.DirichletProcessMixtures.map_assignments(feature_model)
    ll = HybridSpikeSorter.DirichletProcessMixtures.lratio(feature_model, feature_data)
    clusters = unique(cids)
    sort!(clusters)
    nclusters = maximum(clusters)
     _colors = [(cc.r, cc.g, cc.b) for cc in distinguishable_colors(nclusters, colorant"tomato";lchoices=linspace(10,80,15))]
    ax2[:clear]()
    for i in 1:size(spike_model.template_model.μ,2)
        ax2[:plot](spike_model.template_model.μ[:,i];color=_colors[clusterid[i]],label="Cluster $(clusterid[i])")
    end
    ax2[:legend]()
    ax1[:scatter](feature_data[1,:], feature_data[2,:], feature_data[3,:];s=1.0, c=_colors[cids])
    x = collect(keys(ll))
    ax1[:set_xlabel]("PCA 1")
    ax1[:set_ylabel]("PCA 2")
    ax1[:set_zlabel]("PCA 3")
    sort!(x)
    ax4[:bar](x, log.([ll[k] for k in x]);color=_colors[x])
    ax4[:axhline](log(max_lratio);label="L-ratio threshold")
    ax4[:set_xticks](x)
    ax4[:set_ylabel]("log(L-ratio)")
    ax4[:legend]()
    plt[:tight_layout](true)
end

function visualize_clusters(model, Y::Matrix{Float64})
    Ys = Signal(Y)
    cids = Signal(HybridSpikeSorter.DirichletProcessMixtures.map_assignments(model))
    visualize_clusters(Ys,cids)
    Ys,cids
end

function visualize_clusters(Y::Signal{Matrix{Float64}}, cids::Signal{Vector{Int64}})
    window = glscreen("Cluster Pot", resolution=(1024,1024))
    active_clusters = map(cids) do _cids
            clusterids = unique(_cids)
            sort!(clusterids)
            nclusters = maximum(clusterids)
            return ones(UInt8, nclusters)
    end
    _colors = map(Y, cids) do _Y, _cids
        if isempty(_cids)
            cc = [Point3f0(0.4f0, 0.3f0, 0.1f0) for i in 1:size(_Y,2)]
        else
            clusterids = unique(_cids)
            sort!(clusterids)
            nclusters = maximum(clusterids)
            _cluster_colors = distinguishable_colors(nclusters, colorant"tomato";lchoices=linspace(10,80,15))
             cc = Array{Point3f0}(length(_cids))
             for i in 1:length(cc)
                 q = _cluster_colors[_cids[i]]
                 cc[i] = Point3f0(Float32(q.r), Float32(q.g), Float32(q.b))
             end
        end
        cc
    end
    #create the model
    centroid = -mean(value(Y),2)[:]
    pm1, pm2, pm3 = extrema(value(Y),2)
    dotranslate = translationmatrix(Vec3f0(centroid))
    doscale = scalematrix(Vec3f0(1.0/(pm1[2]-pm1[1]), 1.0/(pm2[2] - pm2[1]), 1.0/(pm3[2] - pm3[1])))
    thismodel = doscale*dotranslate

    _points = map(Y) do _Y
        [Point3f0(Float32(_Y[1,i]), Float32(_Y[2,i]), Float32(_Y[3,i])) for i in 1:size(_Y,2)]
    end
    vpoints = visualize((Circle, _points),scale=Vec3f0(0.005), color=_colors,model=thismodel)
    const robj = vpoints.children[]
    const gpu_colors = robj[:color]
    const m2id = GLWindow.mouse2id(window)
    @materialize mouseposition, mouse_buttons_pressed, mouse_button_released = window.inputs
    preserve(map(mouse_button_released) do aa
                 id, index = value(m2id)
                 vcids = value(cids)
                 aclusters = value(active_clusters)
                 if id == robj.id && 0 < index < length(gpu_colors)
                     _cid = vcids[index]
                     for ii in 1:length(gpu_colors)
                         cc = vcids[ii]
                         if cc == _cid
                             if aclusters[_cid] == 1
                                 gpu_colors[ii] += 0.5f0*(Point3f0(1.f0, 1.f0, 1.f0) - gpu_colors[ii])
                             else
                                 gpu_colors[ii] = (gpu_colors[ii] - 0.5f0*Point3f0(1.0f0))/(1.f0 - 0.5f0)
                             end
                         end
                     end
                     if aclusters[_cid] == 1
                         aclusters[_cid] = 0
                     elseif aclusters[_cid] == 0
                         aclusters[_cid] = 1
                     end
                 end
                 push!(active_clusters, aclusters)
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

