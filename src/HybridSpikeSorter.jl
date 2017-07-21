module HybridSpikeSorter
using StatsBase
using ConjugatePriors
using MultivariateStats
using SpikeExtraction
using SpikeSorter
using HMMSpikeSorter
using DirichletProcessMixtures
using RippleTools
using FileIO
using JLD
using Colors

"""
Sort spikes from wide band data recorded at `sampling_rate` on `channel`. Waveforms are extracted as 1.5 ms window are peaks exceeding 6 times the standard deviation of the high pass filtered (500-10kHz). A feature space is created by retaining the first 5 principcal components of the waveforms, and a dirichlet process gaussian mixture model (DPGMM) is fitted to this space using `max_clusters` as the truncation parameter. Clusters with a l-ratio less than `max_lratio` are retained as representing putative single units. Finally, a hidden markov model (HMM) is fit using these units.
"""
function sort_spikes(data::Vector{Float64},sampling_rate::Real,channel::Int64;chunksize=80000,max_clusters=10,max_lratio=20.0,max_iter=1000)
    fdata = SpikeExtraction.highpass_filter(data, sampling_rate) 
    pts = round(Int64,1.5*sampling_rate/1000)
    n1 =div(pts,3)
    n2 = pts-n1 
    μ0, σ0 = SpikeExtraction.get_threshold(fdata)
    idx, waveforms = SpikeExtraction.extract_spikes(fdata;μ=μ0, σ=σ0, nq=(n1,n2), only_negative=true)
    pca = fit(PCA, waveforms;maxoutdim=5)
    y = transform(pca, waveforms)
    C = inv(diagm(pca.prinvars))
    prior = ConjugatePriors.NormalWishart(zeros(5), 1e-7, C, 5.001)
    model = DirichletProcessMixtures.DPGMM(size(y,2),max_clusters, 1e-10, prior;random_init=true);
    niter = DirichletProcessMixtures.infer(model,y, max_iter, 1e-2)
    cids = DirichletProcessMixtures.map_assignments(model)
    ll = DirichletProcessMixtures.lratio(cids,y)
    llm = filter((k,v)->v < max_lratio, ll)
    clusters = collect(keys(llm))
    sort!(clusters)
    μ = cat(2, [mean(waveforms[:,cids.==c],2) for c in clusters]...)
    μ[1,:] = 0.0
    cc = countmap(cids)
    lp = [log(cc[k]/length(fdata)) for k in clusters]
    templates = HMMSpikeSorter.HMMSpikeTemplateModel(μ, lp,true);
    templates.σ = σ0
    modelf = fit(HMMSpikeSorter.HMMSpikingModel, templates, fdata, chunksize)
    JLD.@save "model.jld" modelf
    units = HMMSpikeSorter.extract_units(modelf,channel;sampling_rate=sampling_rate)
    units, model, y, modelf
end

function sort_spikes(datafile::File{format"NSHR"},channel::Int64;kvs...)
    data = RippleTools.get_rawdata(datafile.filename, channel)
    units,model, y, modelf = sort_spikes(data,30_000.0;kvs...)
    JLD.save("$(datafile.filename)_channel$(channel)_sorting_models.jld",Dict("units" => units,
                                                                              "feature_model" => model,
                                                                              "spike_model" => modelf,
                                                                              "feature_data" => y))
end
end #module
