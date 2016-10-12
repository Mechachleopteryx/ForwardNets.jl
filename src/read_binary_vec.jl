function read_binary_vec{T<:Real}(io::IO, ::Type{T})
    vec = T[]
    while !eof(io)
        push!(vec, read(io, T))
    end
    vec
end

function infer_shape{T}(vec::Vector{T}, shape::Tuple{Vararg{Int}})

    #=
    Setting a dimension to -1 will have us infer the other one
    =#

    l = length(vec)

    tot = 1
    dim_to_infer = 0
    for (i,v) in enumerate(shape)
        if v == -1
            @assert(dim_to_infer == 0)
            dim_to_infer = i
        else
            tot *= v
        end
    end

    if dim_to_infer != 0
        @assert(mod(l, tot) == 0)
        dims = collect(shape)
        dims[dim_to_infer] = div(l, tot)
        shape = tuple(dims...)
    end

    shape
end

function _convert_to_column_major_array!{T}(retval::Array{T, 1}, vec::Vector{T}, shape::Tuple{Int})
    copy!(retval, vec)
end
function _convert_to_column_major_array!{T}(retval::Array{T, 2}, vec::Vector{T}, shape::Tuple{Int, Int})
    n, m = shape

    count = 0
    for i in 1 : n
        for j in 1: m
            count += 1
            retval[i,j] = vec[count]
        end
    end

    retval
end
function _convert_to_column_major_array!{T}(retval::Array{T, 3}, vec::Vector{T}, shape::Tuple{Int, Int, Int})
    m, n, o = shape

    count = 0
    for i in 1 : m
        for j in 1 : n
            for k in 1 : o
                count += 1
                retval[i,j, k] = vec[count]
            end
        end
    end

    retval
end
function _convert_to_column_major_array!{T}(retval::Array{T, 4}, vec::Vector{T}, shape::Tuple{Int, Int, Int, Int})
    m, n, o, p = shape

    count = 0
    for i in 1 : m
        for j in 1 : n
            for k in 1 : o
                for l in 1 : p
                    count += 1
                    retval[i,j, k, l] = vec[count]
                end
            end
        end
    end

    retval
end

function convert_to_column_major_array!{T}(retval::Array{T, 1}, vec::Vector{T}, shape::Tuple{Int})
    @assert(shape[1] == length(vec))
    _convert_to_column_major_array(retval, vec, shape)
end
function convert_to_column_major_array!{T}(retval::Array{T, 2}, vec::Vector{T}, shape::Tuple{Int, Int})
    n, m = shape
    @assert(n*m == length(vec))
    @assert(size(retval) == shape)

    _convert_to_column_major_array(retval, vec, shape)
end
function convert_to_column_major_array!{T}(retval::Array{T, 3}, vec::Vector{T}, shape::Tuple{Int, Int, Int})
    m, n, o = shape
    @assert(n*m*o == length(vec))
    @assert(size(retval) == shape)

    _convert_to_column_major_array(retval, vec, shape)
end
function convert_to_column_major_array!{T}(retval::Array{T, 4}, vec::Vector{T}, shape::Tuple{Int, Int, Int, Int})
    m, n, o, p = shape
    @assert(n*m*o*p == length(vec))
    @assert(size(retval) == shape)

    _convert_to_column_major_array(retval, vec, shape)
end

function convert_to_column_major_array{T}(vec::Vector{T}, shape::Tuple{Int})
    @assert(shape[1] == length(vec))
    deepcopy(vec)
end
function convert_to_column_major_array{T}(vec::Vector{T}, shape::Tuple{Int, Int})
    n, m = shape
    @assert(n*m == length(vec))

    retval = Array(T, n, m)

    _convert_to_column_major_array(retval, vec, shape)
end
function convert_to_column_major_array{T}(vec::Vector{T}, shape::Tuple{Int, Int, Int})
    m, n, o = shape
    @assert(n*m*o == length(vec))

    retval = Array(T, m, n, o)

    _convert_to_column_major_array(retval, vec, shape)
end
function convert_to_column_major_array{T}(vec::Vector{T}, shape::Tuple{Int, Int, Int, Int})
    m, n, o, p = shape
    @assert(n*m*o*p == length(vec))

    retval = Array(T, m, n, o, p)

    _convert_to_column_major_array(retval, vec, shape)
end