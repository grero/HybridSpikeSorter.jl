module HybridSpikeSorter
using StatsBase
using ConjugatePriors
using MultivariateStats
using SpikeExtraction
using SpikeSorter
using HMMSpikeSorter
using DirichletProcessMixtures
using RippleTools
using PlexonTools
using FileIO
using JLD
using Colors
using ExperimentDataTools

"""
Sort spikes from wide band data recorded at `sampling_rate` on `channel`. Waveforms are extracted as 1.5 ms window are peaks exceeding 6 times the standard deviation of the high pass filtered (500-10kHz). A feature space is created by retaining the first 5 principcal components of the waveforms, and a dirichlet process gaussian mixture model (DPGMM) is fitted to this space using `max_clusters` as the truncation parameter. Clusters with a l-ratio less than `max_lratio` are retained as representing putative single units. Finally, a hidden markov model (HMM) is fit using these units.
"""
function sort_spikes!(sorted_data::Dict, data::Vector{Float64},sampling_rate::Real,channel::Int64;chunksize=80000,max_clusters=10,max_lratio=20.0,max_iter=1000,fname="",max_restarts=5,min_number_of_spikes=100,time_adjust=Float64[])
    fdata = SpikeExtraction.highpass_filter(data, sampling_rate) 
    pts = round(Int64,1.5*sampling_rate/1000)
    n1 =div(pts,3)
    n2 = pts-n1 
    μ0, σ0 = SpikeExtraction.get_threshold(fdata)
    if !("feature_model" in keys(sorted_data))
        if !isempty(fname)
            pp,ext = splitext(fname)
            if ext != ".jld"
                throw(ArgumentError("Filename $(fname) is not a valid JLD file. It should have extension .JLD"))
            end
        end
        widx, waveforms = SpikeExtraction.extract_spikes(fdata;μ=μ0, σ=σ0, nq=(n1,n2), only_negative=true)
        pca = fit(PCA, waveforms;maxoutdim=5)
        y = transform(pca, waveforms)
        C = inv(diagm(pca.prinvars))
        r = 0
        success = false
        prior = ConjugatePriors.NormalWishart(zeros(5), 1e-7, C, 5.001)
        model = DirichletProcessMixtures.DPGMM(size(y,2),max_clusters, 1e-10, prior;random_init=true);
        try
            niter = DirichletProcessMixtures.infer(model,y, max_iter, 1e-2)
        catch Base.Linalg.SingularException
            success = false
            println("Restarting model fit...")
            r += 1
        end
        success = true
        while !success && r < max_restarts
            try
                prior = ConjugatePriors.NormalWishart(zeros(5), 1e-7, C, 5.001)
                model = DirichletProcessMixtures.DPGMM(size(y,2),max_clusters, 1e-10, prior;random_init=true);
                niter = DirichletProcessMixtures.infer(model,y, max_iter, 1e-2)
            catch Base.Linalg.SingularException
                success = false
                println("Restarting model fit...")
                r += 1
            end
            success = true
        end
        sorted_data["feature_model"] = model
        sorted_data["feature_data"] = y
        sorted_data["waveforms"] = waveforms
        sorted_data["spikeidx"] = widx
        sorted_data["max_clusters"] = max_clusters
        sorted_data["max_lratio"] = max_lratio
        sorted_data["min_number_of_spikes"] = min_number_of_spikes
        sorted_data["sampling_rate"] = sampling_rate
    end
    model = sorted_data["feature_model"]
    y = sorted_data["feature_data"]
    waveforms = sorted_data["waveforms"]
    if !("spike_model" in keys(sorted_data))
        cids = DirichletProcessMixtures.map_assignments(model)
        spike_counts = countmap(cids)
        ll = DirichletProcessMixtures.lratio(cids,y)
        llm = filter((k,v)->(v < max_lratio)&(spike_counts[k] >= min_number_of_spikes), ll)
        clusters = collect(keys(llm))
        if !isempty(clusters) # no clusters fulfilled the l-ratio critera
            sort!(clusters)
            μ = cat(2, [mean(waveforms[:,cids.==c],2) for c in clusters]...)
            μ[1,:] = 0.0
            cc = countmap(cids)
            lp = [log(cc[k]/length(fdata)) for k in clusters]
            templates = HMMSpikeSorter.HMMSpikeTemplateModel(μ, lp,true);
            templates.σ = σ0
            modelf = fit(HMMSpikeSorter.HMMSpikingModel, templates, fdata, chunksize)
            units = HMMSpikeSorter.extract_units(modelf,channel;sampling_rate=sampling_rate)
            save_units(units)
            sorted_data["units"] = units
            sorted_data["spike_model"] = modelf
            sorted_data["clusterid"] = clusters
            sorted_data["max_lratio"] = max_lratio
            sorted_data["min_number_of_spikes"] = min_number_of_spikes
            sorted_data["cid"] = cids
        end
    end
    try
        if !isempty(fname)
            #move the raw data from modelf to its own entry so that we can easily read it from e.g. matlab
            sorted_data["y"] = sorted_data["spike_model"].y
            sorted_data["spike_model"].y = Float64[]
            JLD.save(fname,sorted_data)
            #restore it
            sorted_data["spike_model"].y = sorted_data["y"]
            delete!(sorted_data, "y")
        end
    catch
        warn("Unable to save data")
    end
    sorted_data
end

function sort_spikes(data::Dict,sampling_rate::Real,fname::String;kvs...)
    pmap((k,v)->begin
             _fname = "$(fname)_channel$(k)_sorting_models.jld"
             sort_spikes(v,sampling_rate,k;fname=_fname, kvs...)
         end,
         data)
end

function sort_spikes(datafile::File{format"NSHR"},channel::Int64;kvs...)
    pp,ext = splitext(datafile.filename)
    if !isdir("sorted")
        mkdir("sorted")
    end
    fname = "sorted/$(pp)_channel_$(channel)_sorting.jld"
    println("Results will be saved to $(fname)")
    data = RippleTools.get_rawdata(datafile.filename, channel)
    sorted_data = Dict()
    sorted_data = sort_spikes!(sorted_data, data[channel],30_000.0,channel;fname=fname,kvs...)
end

function sort_spikes(datafile::File{format"PL2"}, channel::Int64;kvs...)
    pp,ext = splitext(datafile.filename)
    if !isdir("sorted")
        mkdir("sorted")
    end
    fname = "sorted/$(pp)_channel_$(channel)_sorting.jld"
    println("Results will be saved to $(fname)")
    ch_str = @sprintf "WB%03d" channel
    ad, ts, fn ,adfreq = PlexonTools.get_rawdata(datafile.filename, ch_str);    
    sorted_data = Dict()
    sorted_data = sort_spikes!(sorted_data, ad, adfreq, channel;fname=fname,kvs...)
    if "units" in keys(sorted_data)
        for (k,v) in sorted_data["units"]
            PlexonTools.adjust_spiketimes!(v["timestamps"],ts,fn,adfreq)
        end
        save_units(sorted_data["units"])
    end
    return sorted_data
end

"""
Load the spike sorting results from the file `f`. The main reason for using this function over JLD.load is that it will reconstruct the spike_model with the raw data, while JLD.load returns a dictionary with the raw data in the variable "y"
"""
function load_sorting(f::File{format"JLD"})
    sorted_data = JLD.load(f)
    if "y" in keys(sorted_data)
        if isempty(sorted_data["spike_model"].y)
            sorted_data["spike_model"].y = sorted_data["y"]
            delete!(sorted_data,"y")
        end
    end
    sorted_data
end

function save_units(units::Dict)
    for (k,v) in units
        @show keys(v)
        ExperimentDataTools.get_session_spiketimes(k,v)
    end
end

end #module
