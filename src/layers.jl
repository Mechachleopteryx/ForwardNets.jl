type Affine{T} <: Layer{T}
    name::Symbol
    W::Matrix{T} # o×i
    b::Vector{T} # o

    input::Vector{T} # i
    output::Vector{T}  # o
end
name(a::Affine) = a.name
output(a::Affine) = a.output
function forward!{T}(a::Affine{T})
    copy!(a.output, a.b) # y ← b
    Base.LinAlg.BLAS.gemv!('N', one(T), a.W, a.input, one(T), a.output) # y ← W*x + y
    a
end
function Base.push!{T}(net::ForwardNet{T}, ::Type{Affine},
    name::Symbol,
    parent::NameOrIndex,
    output_dim::Int,
    )

    input = ForwardNets.output(net[parent])::Vector{T}

    W = Array(T, output_dim, length(input))
    b = Array(T, output_dim)
    output = Array(T, output_dim)

    node = Affine(name, W, b, input, output)
    push!(net, node, parent)
end
Base.push!{T}(net::ForwardNet{T}, ::Type{Affine}, parent::NameOrIndex, output_dim::Int) =
    push!(net, Affine, create_next_name!(net, "Affine"), parent, output_dim)
function restore!{T}(a::Affine{T}, filename_W::String, filename_b::String)
    input_dim = length(a.input)

    vec = open(io->read_binary_vec(io, T), filename_W)
    shape = infer_shape(vec, (-1, input_dim))
    a.W[:] = convert_to_column_major_array(vec, (shape[2], shape[1]))'

    copy!(a.b, open(read_binary_vec, filename_b))

    a
end


type Concatenator{T} <: Layer{T}
    name::Symbol
    inputs::Vector{Vector{T}}
    output::Vector{T}
end
name(a::Concatenator) = a.name
output(a::Concatenator) = a.output
function forward!(a::Concatenator)
    i = 0
    for input in a.inputs
        for j in 1 : length(input)
            i += 1
            a.output[i] = input[j]
        end
    end
    a
end
function Base.push!{T, V<:NameOrIndex}(net::ForwardNet{T}, ::Type{Concatenator},
    name::Symbol,
    parents::Vector{V},
    )

    tot_len = 0
    inputs = Array(Vector{T}, length(parents))
    for (i,parent) in enumerate(parents)
        inputs[i] = ForwardNets.output(net[parent])::Vector{T}
        tot_len += length(inputs[i])
    end

    output = Array(T, tot_len)

    node = Concatenator{T}(name, inputs, output)
    push!(net, node, parents)
end
Base.push!{T, V<:NameOrIndex}(net::ForwardNet{T}, ::Type{Concatenator}, parents::Vector{V}) =
    push!(net, Concatenator, create_next_name!(net, "Concatenator"), parents)

type Reshaper{T} <: Layer{T}
    name::Symbol
    input::Array{T}
    output::Array{T}
end
name(a::Reshaper) = a.name
output(a::Reshaper) = a.output
function forward!(a::Reshaper)
    dest_ind = 0
    if ndims(a.input) == 1
        copy!(a.output, a.input)
    elseif ndims(a.input) == 2
        for i in 1 : size(a.input, 1), j in 1 : size(a.input, 2)
            a.output[dest_ind+=1] = a.input[i,j]
        end
    elseif ndims(a.input) == 3
        for i in 1 : size(a.input, 1), j in 1 : size(a.input, 2), k in 1 : size(a.input, 3)
            a.output[dest_ind+=1] = a.input[i,j,k]
        end
    elseif ndims(a.parent) == 4
        for i in 1 : size(a.input, 1), j in 1 : size(a.input, 2), k in 1 : size(a.input, 3), h in 1:size(a.input,4)
            a.output[dest_ind+=1] = a.input[i,j,k,h]
        end
    end
    a
end
function Base.push!{T}(net::ForwardNet{T}, ::Type{Reshaper},
    name::Symbol,
    parent::NameOrIndex,
    new_shape::Tuple{Vararg{Int}},
    )

    input = ForwardNets.output(net[parent])::Array{T}
    output = Array(T, new_shape...)

    node = Reshaper(name, input, output)
    push!(net, node, parent)
end
Base.push!{T}(net::ForwardNet{T}, ::Type{Reshaper}, parent::NameOrIndex, new_shape::Tuple{Vararg{Int}},) =
    push!(net, Reshaper, create_next_name!(net, "Reshaper"), parent, new_shape)