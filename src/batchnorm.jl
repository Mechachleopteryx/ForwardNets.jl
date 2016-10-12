type BatchNorm1{T} <: Layer{T}
    # https://arxiv.org/pdf/1502.03167.pdf
    name::Symbol
    γ::Vector{T}
    β::Vector{T}
    ϵ::T
    μ::Vector{T} # train set mean
    ν::Vector{T} # train set variance

    input::Vector{T}
    output::Vector{T}
end
name(a::BatchNorm1) = a.name
output(a::BatchNorm1) = a.output
function forward!(a::BatchNorm1)
    x = (a.input - a.μ) ./ sqrt(a.ν + a.ϵ)
    a.output .= (a.γ .* x) .+ a.β
    a
end
function Base.push!{T}(net::ForwardNet{T}, ::Type{BatchNorm1},
    name::Symbol,
    parent::NameOrIndex,
    ϵ::T=T(1e-5)
    )

    input = output(net[parent])::Vector{T}
    output = Array(T, length(input))
    γ = Array(T, length(input))
    β = Array(T, length(input))
    μ = Array(T, length(input))
    ν = Array(T, length(input))

    node = BatchNorm1(name, γ, β, ϵ, μ, ν, input, output)
    push!(net, node, parent)
end
Base.push!{T}(net::ForwardNet{T}, ::Type{BatchNorm1}, parent::NameOrIndex, ϵ::T=T(1e-5)) =
    push!(net, BatchNorm1, create_next_name!(net, "BatchNorm1"), parent, ϵ)
function restore!{T}(a::BatchNorm1{T},
    filename_gamma::String,
    filename_beta::String,
    filename_mean::String,
    filename_variance::String)

    copy!(a.γ, open(io->read_binary_vec(io, T), filename_gamma))
    copy!(a.β, open(io->read_binary_vec(io, T), filename_beta))
    copy!(a.μ, open(io->read_binary_vec(io, T), filename_mean))
    copy!(a.ν, open(io->read_binary_vec(io, T), filename_variance))

    a
end

type BatchNorm3{T} <: Layer{T}
    # https://arxiv.org/pdf/1502.03167.pdf
    name::Symbol
    γ::Vector{T}
    β::Vector{T}
    ϵ::T
    μ::Vector{T} # train set mean
    ν::Vector{T} # train set variance

    input::Array{T, 3}
    output::Array{T, 3}
end
name(a::BatchNorm3) = a.name
output(a::BatchNorm3) = a.output
function forward!(a::BatchNorm3)
    for i in 1 : size(a.output)[3]
        x = (a.input[:, :, i] - a.μ[i]) / sqrt(a.ν[i] + a.ϵ)
        a.output[:, :, i] = a.γ[i] * x + a.β[i]
    end
    a
end
function Base.push!{T}(net::ForwardNet{T}, ::Type{BatchNorm3},
    name::Symbol,
    parent::NameOrIndex,
    ϵ::T=T(1e-5)
    )

    input = output(net[parent_index])::Array{T, 3}
    output = Array(T, size(input))
    in_c = size(input, 3)
    γ = Array(T, in_c)
    β = Array(T, in_c)
    μ = Array(T, in_c)
    ν = Array(T, in_c)

    node = BatchNorm3(name, γ, β, ϵ, μ, ν, input, output)
    push!(net, node, parent)
end
Base.push!{T}(net::ForwardNet{T}, ::Type{BatchNorm3}, parent::NameOrIndex, ϵ::T=T(1e-5)) =
    push!(net, BatchNorm3, create_next_name!(net, "BatchNorm3"), parent, ϵ)
function restore!{T}(a::BatchNorm3{T},
    filename_gamma::String,
    filename_beta::String,
    filename_mean::String,
    filename_variance::String)

    copy!(a.γ, open(io->read_binary_vec(io, T), filename_gamma))
    copy!(a.β, open(io->read_binary_vec(io, T), filename_beta))
    copy!(a.μ, open(io->read_binary_vec(io, T), filename_mean))
    copy!(a.ν, open(io->read_binary_vec(io, T), filename_variance))

    a
end