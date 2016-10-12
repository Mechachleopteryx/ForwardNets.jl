add_node!{T}(net::ForwardNet{T}, node::Node{T}, parents::Vector{Int}=Int[]) = push!(net, node, parents)
add_node!{T}(net::ForwardNet{T}, node::Node{T}, parent::Int) = push!(net, node, Int[parent])