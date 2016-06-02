VERSION >= v"0.4.0-dev+6521" && __precompile__(true)

module ForwardNets

using LightGraphs

export
    ForwardNet,
    Node,
    Layer,
    Variable,
    Activation,
    Affine,
    LSTM,
    Conv2d,
    BatchNorm,
    Concatenator,
    Reshaper,
    ReLU,
    SoftPlus,
    ForwardPass,

    read_binary_vec,
    infer_shape,
    convert_to_column_major_array,

    forward!,
    add_node!,
    restore!,
    indexof,
    lastindex,
    get_name,
    get_output,

    sigmoid,
    sigmoid!,
    tanh!,

    zero!,

    calc_forwardpass,
    print_forward_pass


include("read_binary_vec.jl")

abstract Node

const DEFAULT_NAME = :unnamed
function forward!(n::Node)
    # do nothing by default
end
get_name(n::Node) = DEFAULT_NAME

#############################################

type ForwardNet
    dag::DiGraph # graph over nodes
    nodes::Vector{Node}
    name_to_index::Dict{Symbol,Int} # Symbol → index in dag and nodes (note: not all Nodes have names)

    ForwardNet() = new(DiGraph(0), Node[], Dict{Symbol,Int}())
end

Base.getindex(net::ForwardNet, index::Int) = net.nodes[index]
Base.getindex(net::ForwardNet, name::Symbol) = net.nodes[net.name_to_index[name]]
indexof(net::ForwardNet, name::Symbol) = net.name_to_index[name]
lastindex(net::ForwardNet) = nv(net.dag)
function add_node!(net::ForwardNet, node::Node, parents::Vector{Int}=Int[])
    add_vertex!(net.dag)
    push!(net.nodes, node)
    i = length(net.nodes)

    name = get_name(node)
    if name != DEFAULT_NAME
        @assert(!haskey(net.name_to_index, name))
        net.name_to_index[name] = i
    end

    for parent in parents
        add_edge!(net.dag, parent, i)
    end

    net
end
add_node!(net::ForwardNet, node::Node, parent::Int) = add_node!(net, node, Int[parent])

#############################################

abstract Layer <: Node
abstract Activation <: Node

type Variable{n} <: Node
    name::Symbol
    tensor::Array{Float32, n}
end
get_name(a::Variable) = a.name
get_output(a::Variable) = a.tensor

type Affine <: Layer
    name::Symbol
    W::Matrix{Float32} # o×i
    b::Vector{Float32} # o

    parent::Vector{Float32} # i
    child::Vector{Float32}  # o
end
get_name(a::Affine) = a.name
get_output(a::Affine) = a.child
function forward!(a::Affine)
    copy!(a.child, a.b) # y ← b
    Base.LinAlg.BLAS.gemv!('N', 1.0f0, a.W, a.parent, 1.0f0, a.child) # y ← W*x + y
    a
end
function add_node!(net::ForwardNet, ::Type{Affine},
    name::Symbol,
    parent_index::Int,
    output_dim::Int,
    )

    parent_node = net[parent_index]

    parent = get_output(parent_node)::Vector{Float32}

    W = Array(Float32, output_dim, length(parent))
    b = Array(Float32, output_dim)
    child = Array(Float32, output_dim)

    node = Affine(name, W, b, parent, child)
    add_node!(net, node, parent_index)
end
function restore!(a::Affine, filename_W::AbstractString, filename_b::AbstractString)
    input_dim = length(a.parent)

    vec = open(read_binary_vec, filename_W)
    shape = infer_shape(vec, (-1, input_dim))
    a.W[:] = convert_to_column_major_array(vec, (shape[2], shape[1]))'

    copy!(a.b, open(read_binary_vec, filename_b))

    a
end

sigmoid(x::Real) = 1 / (1 + exp(-x))
function sigmoid{F<:Real}(x::Vector{F})
    retval = deepcopy(x)
    for i in 1 : length(x)
        retval[i] = sigmoid(x[i])
    end
    retval
end
function sigmoid!{F<:Real}(x::Vector{F})
    for i in 1 : length(x)
        x[i] = sigmoid(x[i])
    end
    x
end
function tanh!{F<:Real}(x::Vector{F})
    for i in 1 : length(x)
        x[i] = tanh(x[i])
    end
    x
end

type LSTM <: Layer
    name::Symbol
    W::Matrix{Float32} # 4H×(D+H)
        # consists of Wx and Wh
        #  Wx: Input-to-hidden weights, of shape (4H, D)
        #  Wh: Hidden-to-hidden weights, of shape (4H, H)
    b::Vector{Float32} # 4H
    forget_bias::Float32

    parent::Vector{Float32} # input, [D]
    state::Vector{Float32} # state (2H), contains [cell state, hidden state]

    # preallocated memory
    c::Vector{Float32} # [H] cell state
    m::Vector{Float32} # [H] member state
    cell_inputs::Vector{Float32}  # [D+H]
    lstm_mult_res::Vector{Float32} # [4H]
    i::Vector{Float32} # [H] input gate
    j::Vector{Float32} # [H] new input
    f::Vector{Float32} # [H] forget gate
    o::Vector{Float32} # [H] output gate
end
get_name(a::LSTM) = a.name
get_output(a::LSTM) = a.m
zero!(a::LSTM) = fill!(a.state, 0.0f0)
function add_node!(net::ForwardNet, ::Type{LSTM},
    name::Symbol,
    parent_index::Int,
    H::Int; # hidden layer size
    forget_bias::Float32 = 1.0f0
    )

    parent = get_output(net[parent_index])::Vector{Float32}
    D = length(parent)

    W = Array(Float32, 4H, D+H)
    b = Array(Float32, 4H)
    state = Array(Float32, 2H)

    c = Array(Float32, H)
    m = Array(Float32, H)
    cell_inputs = Array(Float32, D+H)
    lstm_mult_res = Array(Float32, 4H)
    i = Array(Float32, H)
    j = Array(Float32, H)
    f = Array(Float32, H)
    o = Array(Float32, H)

    node = LSTM(name, W, b, forget_bias, parent, state, c, m, cell_inputs, lstm_mult_res, i, j, f, o)
    add_node!(net, node, parent_index)
end
function forward!(a::LSTM)

    H = length(a.c)
    D = length(a.parent)

    copy!(a.c, 1, a.state, 1, H)
    copy!(a.m, 1, a.state, H+1, H)

    copy!(a.cell_inputs, 1, a.parent, 1, D)
    copy!(a.cell_inputs, D+1, a.m, 1, H)

    # lstm_matrix = a.W * a.cell_inputs + a.b
    copy!(a.lstm_mult_res, a.b) # y ← b
    Base.LinAlg.BLAS.gemv!('N', 1.0f0, a.W, a.cell_inputs, 1.0f0, a.lstm_mult_res) # y ← W*x + y

    copy!(a.i, 1, a.lstm_mult_res,    1, H)
    copy!(a.j, 1, a.lstm_mult_res,  H+1, H)
    copy!(a.f, 1, a.lstm_mult_res, 2H+1, H)
    copy!(a.o, 1, a.lstm_mult_res, 3H+1, H)

    # compute c_next and m_next
    for k in 1 : H
        a.f[k] += a.forget_bias
        c = sigmoid(a.f[k])*a.c[k] + sigmoid(a.i[k])*tanh(a.j[k]) # c_next
        a.state[k] = a.c[k] = c
        a.state[k+H] = a.m[k] = sigmoid(a.o[k])*tanh(c) # m_next
    end

    a
end
function restore!(a::LSTM, filename_W::AbstractString, filename_b::AbstractString)
    H = length(a.c)
    D = length(a.parent)

    vec = open(read_binary_vec, filename_W)
    a.W[:] = convert_to_column_major_array(vec, (D+H,4H))'

    copy!(a.b, open(read_binary_vec, filename_b))

    a
end

type Conv2d <: Layer
    name::Symbol
    W::Array{Float32, 4} # filter_h × filter_w × in_c × out_c
    b::Vector{Float32} # out_c
    strides::Tuple{Int, Int} # width, height

    parent::Array{Float32, 3} # h × w × in_c
    child::Array{Float32, 3}  # out_h × out_w × out_c
end
get_name(a::Conv2d) = a.name
get_output(a::Conv2d) = a.child
function add_node!(net::ForwardNet, ::Type{Conv2d},
    name::Symbol,
    parent_index::Int,
    strides::Tuple{Int, Int},
    filter_h::Int,
    filter_w::Int,
    out_c::Int)

    parent = get_output(net[parent_index])::Array{Float32, 3}
    h, w, in_c = size(parent)

    W = Array(Float32, filter_h, filter_w, in_c, out_c)
    b = Array(Float32, out_c)

    # FOR VALID: (padding is always zero)
    # out_w = ceil(float(w - filter_w + 1) / float(strides[2]))
    # out_h = ceil(float(h - filter_h + 1) / float(strides[1]))

    # FOR SAME:
    out_w  =ceil(Int, w / strides[2])
    out_h = ceil(Int, h / strides[1])
    child = Array(Float32, out_h, out_w, out_c)

    node = Conv2d(name, W, b, strides, parent, child)
    add_node!(net, node, parent_index)
end
function forward!(a::Conv2d)

    h_in = size(a.parent, 1)
    h_out = size(a.child, 1)
    w_in = size(a.parent, 2)
    w_out = size(a.child, 2)
    hh, ww, f_in, f_out = size(a.W)

    s₁, s₂ = a.strides
    pad_h = (s₁ * (h_out - 1) + hh - h_in) / 2
    pad_w = (s₂ * (w_out - 1) + ww - w_in) / 2

    pad_h_lo = floor(Int, pad_h)
    pad_h_hi = ceil(Int, pad_h)
    pad_w_lo = floor(Int, pad_w)
    pad_w_hi = ceil(Int, pad_w)

    # println("parent_size: ", size(a.parent))
    # println("child_size:  ", size(a.child))

    # println("padding: ", pad_h_lo, "  ", pad_h_hi)
    # println("         ", pad_w_lo, "  ", pad_w_hi)

    i_out = 1
    for i_out in 1 : size(a.child, 1)
        i_in_lo = (i_out-1)*s₁ - pad_h_lo + 1
        i_in_hi = i_in_lo + hh - 1

        h_in_lo = 1
        if i_in_lo < 1
            h_in_lo += 1-i_in_lo
            i_in_lo = 1
        end
        h_in_hi = hh
        if i_in_hi > h_in
            h_in_hi -= i_in_hi - h_in
            i_in_hi = h_in
        end

        for j_out in 1 : size(a.child, 2)

            j_in_lo = (j_out-1)*s₂ - pad_w_lo + 1
            j_in_hi = j_in_lo + ww - 1

            # println("j_in_lo: ", j_in_lo)
            # println("j_in_hi: ", j_in_hi)

            w_in_lo = 1
            if j_in_lo < 1
                w_in_lo += 1-j_in_lo
                j_in_lo = 1
            end
            w_in_hi = ww
            if j_in_hi > w_in
                w_in_hi -= j_in_hi - w_in
                j_in_hi = w_in
            end

            # println("j_in_lo: ", j_in_lo)
            # println("j_in_hi: ", j_in_hi)
            # println("w_in_lo: ", w_in_lo)
            # println("w_in_hi: ", w_in_hi)

            for k_out in 1 : f_out

                # println("parent: ", i_in_lo:i_in_hi, ", ", j_in_lo:j_in_hi, ", ", 1:f_in)
                # println("W:      ", h_in_lo:h_in_hi, ", ", w_in_lo,w_in_hi, ", ", 1:f_in, ", ", k_out)

                a.child[i_out,j_out,k_out] = a.b[k_out]

                for (i, i_in) in enumerate(i_in_lo:i_in_hi)
                    h₁ = h_in_lo + i - 1
                    for (j, j_in) in enumerate(j_in_lo:j_in_hi)
                        w₁ = w_in_lo + j - 1
                        for k_in in 1 : f_in
                            W_contrib = a.W[h₁,w₁,k_in,k_out]
                            p_contrib = a.parent[i_in,j_in,k_in]
                            a.child[i_out,j_out,k_out] += W_contrib*p_contrib
                        end
                    end
                end
            end
        end
    end

    a
end
function restore!(a::Conv2d, filename_W::AbstractString, filename_b::AbstractString)

    filter_h, filter_w, in_c, out_c = size(a.W)

    vec = open(read_binary_vec, filename_W)
    a.W[:] = convert_to_column_major_array(vec, (filter_h, filter_w, in_c, out_c))

    copy!(a.b, open(read_binary_vec, filename_b))

    a
end

type BatchNorm1 <: Layer
    # https://arxiv.org/pdf/1502.03167.pdf
    name::Symbol
    γ::Float32
    β::Float32
    ϵ::Float32
    μ::Float32 # train set mean
    ν::Float32 # train set variance

    parent::Vector{Float32}
    child::Vector{Float32}
end
get_name(a::BatchNorm1) = a.name
get_output(a::BatchNorm1) = a.child
function forward!(a::BatchNorm1)
    x = (a.parent - a.μ) / sqrt(a.ν + a.ϵ)
    a.child = a.γ * x + a.β
    a
end
function add_node!(net::ForwardNet, ::Type{BatchNorm1},
    name::Symbol,
    parent_index::Int,
    ϵ::Float32=Float32(1e-5)
    )

    parent_node = net[parent_index]

    parent = get_output(parent_node)::Vector{Float32}
    child = Array(Float32, length(parent))
    γ = Float32
    β = Float32
    μ = Float32
    ν = Float32

    node = BatchNorm1(name, γ, β, ϵ, μ, ν, parent, child)
    add_node!(net, node, parent_index)
end
function restore!(a::BatchNorm1, filename_gamma::AbstractString, filename_beta::AbstractString,
    filename_mean::AbstractString,
    filename_variance::AbstractString)

    copy!(a.γ, open(read_binary_vec, filename_gamma))
    copy!(a.β, open(read_binary_vec, filename_beta))
    copy!(a.μ, open(read_binary_vec, filename_mean))
    copy!(a.ν, open(read_binary_vec, filename_variance))

    a
end

type BatchNorm3 <: Layer
    # https://arxiv.org/pdf/1502.03167.pdf
    name::Symbol
    γ::Vector{Float32}
    β::Vector{Float32}
    ϵ::Float32
    μ::Vector{Float32} # train set mean
    ν::Vector{Float32} # train set variance

    parent::Array{Float32, 3}
    child::Array{Float32, 3}
end
get_name(a::BatchNorm3) = a.name
get_output(a::BatchNorm3) = a.child
function forward!(a::BatchNorm3)
    for i in 1 : size(a.child)[3]
        x = (a.parent[:, :, i] - a.μ[i]) / sqrt(a.ν[i] + a.ϵ)
        a.child[:, :, i] = a.γ[i] * x + a.β[i]
    end
    a
end
function add_node!(net::ForwardNet, ::Type{BatchNorm3},
    name::Symbol,
    parent_index::Int,
    ϵ::Float32=Float32(1e-5)
    )

    parent_node = net[parent_index]

    parent = get_output(parent_node)::Array{Float32, 3}
    child = Array(Float32, size(parent))
    _, _, in_c = size(parent)
    γ = Array(Float32, in_c)
    β = Array(Float32, in_c)
    μ = Array(Float32, in_c)
    ν = Array(Float32, in_c)

    node = BatchNorm(name, γ, β, ϵ, μ, ν, parent, child)
    add_node!(net, node, parent_index)
end
function restore!(a::BatchNorm3, filename_gamma::AbstractString, filename_beta::AbstractString,
    filename_mean::AbstractString,
    filename_variance::AbstractString)

    copy!(a.γ, open(read_binary_vec, filename_gamma))
    copy!(a.β, open(read_binary_vec, filename_beta))
    copy!(a.μ, open(read_binary_vec, filename_mean))
    copy!(a.ν, open(read_binary_vec, filename_variance))

    a
end

type Concatenator <: Layer
    name::Symbol
    parents::Vector{Vector{Float32}}
    child::Vector{Float32}
end
get_name(a::Concatenator) = a.name
get_output(a::Concatenator) = a.child
function forward!(a::Concatenator)
    i = 0
    for parent in a.parents
        for j in 1 : length(parent)
            i += 1
            a.child[i] = parent[j]
        end
    end
    a
end
function add_node!(net::ForwardNet, ::Type{Concatenator},
    name::Symbol,
    parent_indeces::Vector{Int}
    )

    tot_len = 0
    parents = Array(Vector{Float32}, length(parent_indeces))
    for (i,parent_index) in enumerate(parent_indeces)
        parents[i] = get_output(net[parent_index])::Vector{Float32}
        tot_len += length(parents[i])
    end

    child = Array(Float32, tot_len)

    node = Concatenator(name, parents, child)
    add_node!(net, node, parent_indeces)
end

type Reshaper <: Layer
    parent::Array{Float32}
    child::Array{Float32}
end
get_output(a::Reshaper) = a.child
function forward!(a::Reshaper)
    # copy!(a.child, a.parent)
    dest_ind = 0
    for i in 1 : size(a.parent, 1)
        for j in 1 : size(a.parent, 2)
            for k in 1 : size(a.parent, 3)
                a.child[dest_ind+=1] = a.parent[i,j,k]
            end
        end
    end
    a
end
function add_node!(net::ForwardNet, ::Type{Reshaper},
    parent_index::Int,
    new_shape::Tuple{Vararg{Int}},
    )

    parent = get_output(net[parent_index])::Array{Float32}
    child = Array(Float32, new_shape...)

    node = Reshaper(parent, child)
    add_node!(net, node, parent_index)
end
function add_node!(net::ForwardNet, ::Type{Reshaper},
    parent_index::Int,
    )

    # this just makes it flat

    parent = get_output(net[parent_index])::Array{Float32}
    child = Array(Float32, length(parent))

    node = Reshaper(parent, child)
    add_node!(net, node, parent_index)
end


type ReLU <: Activation
    member::Array{Float32}
end
get_output(a::ReLU) = a.member
function forward!(a::ReLU)
    for i in 1 : length(a.member)
        if a.member[i] < 0.0
            a.member[i] = 0.0
        end
    end
    a
end

type SoftPlus <: Activation
    member::Array{Float32}
end
get_output(a::SoftPlus) = a.member
function forward!(a::SoftPlus)
    for i in 1 : length(a.member)
        a.member[i] = ln(1+exp(a.member[i]))
    end
end

#############################################

immutable ForwardPass
    net::ForwardNet
    input::Array{Symbol}
    output::Array{Symbol}
    activation_order::Vector{Int} # in order
end
function Base.show(io::IO, forwardpass::ForwardPass)
    net = forwardpass.net

    if isempty(forwardpass.activation_order)
        print(io, "ForwardPass: (empty) ", forwardpass.input, " → ", forwardpass.output)
    else
        print(io, "ForwardPass: ", get_name(net[forwardpass.activation_order[1]]))
        for i in 2 : length(forwardpass.activation_order)
            print(io, " → ", get_name(net[forwardpass.activation_order[i]]))
        end
    end
end

function calc_forwardpass(net::ForwardNet, input::Array{Symbol}, output::Array{Symbol})
    #=
    1 - get a topological sort of net
    2 - run through nodes in topologogical order and activate those that are:
            # have at least one ancestor in input and one descendent in output
    3 - will need to run forward! on all active nodes, in topological order
    =#


    input_indeces = Set{Int}()
    input_dijkstra = Dict{Int, LightGraphs.DijkstraState{Int}}()
    for name in input
        index = indexof(net, name)
        push!(input_indeces, index)
        input_dijkstra[index] = dijkstra_shortest_paths(net.dag, index)
    end

    output_indeces = Set{Int}()
    for name in output
        push!(output_indeces, indexof(net, name))
    end

    activation_order = Int[]
    for i in topological_sort_by_dfs(net.dag)

        if !isa(net[i], Variable)

            add_it = false

            for parent in input_indeces

                if parent == i
                    add_it = true
                elseif !in(i, output_indeces) && input_dijkstra[parent].dists[i] != typemax(Int) # is descendent of parent
                    dijkstra = dijkstra_shortest_paths(net.dag, i)

                    for child in output_indeces
                        if dijkstra.dists[child] != typemax(Int)
                            # child is a descendent of our node
                            add_it = true
                        end
                    end

                end
            end

            if add_it
                push!(activation_order, i)
            end
        end
    end

    ForwardPass(net, input, output, activation_order)
end
function forward!(forwardpass::ForwardPass)
    for index in forwardpass.activation_order
        forward!(forwardpass.net.nodes[index])
    end
    forwardpass
end
function print_forward_pass(io::IO, forwardpass::ForwardPass)
    for (count,index) in enumerate(forwardpass.activation_order)
        node = net[index]
        @printf(io, "node%d = net[%d]::%s # %s \n", count, index, typeof(node), get_name(node))
        @printf(io, "forward!(node%d)\n", count)
    end
end
print_forward_pass(forwardpass::ForwardPass) = print_forward_pass(STDOUT, forwardpass)

end # module