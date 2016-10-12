type ForwardNet{T<:Real}
    dag::DiGraph # graph over nodes
    nodes::Vector{Node{T}}
    name_to_index::Dict{Symbol,Int} # Symbol → index in dag and nodes (note: not all Nodes have names)
    name_counter::Int

    ForwardNet() = new(DiGraph(0), Node[], Dict{Symbol,Int}(), 0)
end

Base.getindex(net::ForwardNet, index::Int) = net.nodes[index]
Base.getindex(net::ForwardNet, name::Symbol) = net.nodes[net.name_to_index[name]]
indexof(net::ForwardNet, name::Symbol) = net.name_to_index[name]
lastindex(net::ForwardNet) = nv(net.dag)
function Base.push!{T}(net::ForwardNet{T}, node::Node{T}, parents::Vector{Int}=Int[])

    add_vertex!(net.dag)
    push!(net.nodes, node)
    i = length(net.nodes)

    nodename = ForwardNets.name(node)
    @assert(!haskey(net.name_to_index, nodename))
    net.name_to_index[nodename] = i

    for parent in parents
        add_edge!(net.dag, parent, i)
    end

    net
end
function Base.push!{T}(net::ForwardNet{T}, node::Node{T}, parents::Union{Vector{Symbol}, Vector{NameOrIndex}})
    int_parents = Array(Int, length(parents))
    for (i,p) in enumerate(parents)
        if isa(p, Int)
            int_parents[i] = p
        elseif isa(p, Symbol)
            int_parents[i] = net.name_to_index[p]
        end
    end

    push!(net, node, int_parents)
end
Base.push!{T}(net::ForwardNet{T}, node::Node{T}, parent::Int) = push!(net, node, Int[parent])
Base.push!{T}(net::ForwardNet{T}, node::Node{T}, parent::Symbol) = push!(net, node, Int[net.name_to_index[parent]])

function next_name_number!(net::ForwardNet)
    n = net.name_counter += 1
    n
end
create_next_name!(net::ForwardNet, prefix::String) = convert(Symbol, prefix * "_" * string(next_name_number!(net)))