
type Variable{T,N} <: Node{T}
    name::Symbol
    tensor::Array{T, N}
end
Variable{T}(name::Symbol, ::Type{T}, shape::Int...) = Variable(name, Array(T, shape...))

name(a::Variable) = a.name
output(a::Variable) = a.tensor

function Base.push!{T}(net::ForwardNet{T}, ::Type{Variable},
    name::Symbol,
    tensor::Array{T},
    )

    push!(net, Variable(name, tensor))
end
function Base.push!{T}(net::ForwardNet{T}, ::Type{Variable},
    name::Symbol,
    shape::Int...
    )

    push!(net, Variable(name, Array(T, shape...)))
end
function Base.push!{T}(net::ForwardNet{T}, ::Type{Variable},
    name::Symbol,
    parent::NameOrIndex,
    )

    node = Variable(name, output(net[parent]))
    push!(net, node, parent)
end
