sigmoid(x::Real) = 1 / (1 + exp(-x))
relu(x::Real) = max(x, 0.0)
softplus(x::Real) = log(1 + exp(x))
elu(x::Real) = max(x, 0.0) + (exp(x) - 1)*(x <= 0.0)

function Base.push!{T, A<:Activation}(net::ForwardNet{T}, ::Type{A},
    name::Symbol,
    parent::NameOrIndex,
    )

    input = ForwardNets.output(net[parent])
    output = deepcopy(input)
    node = A(name, input, output)
    push!(net, node, parent)
end
Base.push!{T, A<:Activation}(net::ForwardNet{T}, ::Type{A}, parent::NameOrIndex) =
    push!(net, A, create_next_name!(net, string(A)), parent)

type TanH{T} <: Activation{T}
    name::Symbol
    input::Array{T}
    output::Array{T}
end
name(a::TanH) = a.name
output(a::TanH) = a.output
function forward!(a::TanH)
    a.output .= tanh.(a.input)
    a
end

type ReLU{T} <: Activation{T}
    name::Symbol
    input::Array{T}
    output::Array{T}
end
name(a::ReLU) = a.name
output(a::ReLU) = a.output
function forward!{T}(a::ReLU{T})
    a.output .= max.(a.input, zero(T))
    a
end

type ELU{T} <: Activation{T}
    name::Symbol
    input::Array{T}
    output::Array{T}
end
name(a::ELU) = a.name
output(a::ELU) = a.output
function forward!{T}(a::ELU{T})
    a.output .= elu.(a.input)
    a
end

type SoftPlus{T} <: Activation{T}
    name::Symbol
    input::Array{T}
    output::Array{T}
end
name(a::SoftPlus) = a.name
output(a::SoftPlus) = a.output
function forward!(a::SoftPlus)
    a.output .= softplus.(a.input)
    a
end
