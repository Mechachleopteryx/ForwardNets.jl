type LSTM{T} <: Layer{T}
    name::Symbol
    W::Matrix{T} # 4H×(D+H)
        # consists of Wx and Wh
        #  Wx: Input-to-hidden weights, of shape (4H, D)
        #  Wh: Hidden-to-hidden weights, of shape (4H, H)
    b::Vector{T} # 4H
    forget_bias::T

    input::Vector{T} # input, [D]
    state::Vector{T} # state (2H), contains [cell state, hidden state]

    # preallocated memory
    c::Vector{T} # [H] cell state
    m::Vector{T} # [H] member state
    cell_inputs::Vector{T}  # [D+H]
    lstm_mult_res::Vector{T} # [4H]
    i::Vector{T} # [H] input gate
    j::Vector{T} # [H] new input
    f::Vector{T} # [H] forget gate
    o::Vector{T} # [H] output gate
end
name(a::LSTM) = a.name
output{T}(a::LSTM{T}) = a.m
zero!{T}(a::LSTM{T}) = fill!(a.state, zero(T))
function Base.push!{T}(net::ForwardNet{T}, ::Type{LSTM},
    name::Symbol,
    parent_index::NameOrIndex,
    H::Int; # hidden layer size
    forget_bias::T = one(T),
    )

    input = output(net[parent_index])::Vector{T}
    D = length(input)

    W = Array(T, 4H, D+H)
    b = Array(T, 4H)
    state = Array(T, 2H)

    c = Array(T, H)
    m = Array(T, H)
    cell_inputs = Array(T, D+H)
    lstm_mult_res = Array(T, 4H)
    i = Array(T, H)
    j = Array(T, H)
    f = Array(T, H)
    o = Array(T, H)

    node = LSTM(name, W, b, forget_bias, input, state, c, m, cell_inputs, lstm_mult_res, i, j, f, o)
    push!(net, node, parent_index)
end
function forward!{T}(a::LSTM{T})

    H = length(a.c)
    D = length(a.input)

    copy!(a.c, 1, a.state, 1, H)
    copy!(a.m, 1, a.state, H+1, H)

    copy!(a.cell_inputs, 1, a.input, 1, D)
    copy!(a.cell_inputs, D+1, a.m, 1, H)

    # lstm_matrix = a.W * a.cell_inputs + a.b
    copy!(a.lstm_mult_res, a.b) # y ← b
    Base.LinAlg.BLAS.gemv!('N', one(T), a.W, a.cell_inputs, one(T), a.lstm_mult_res) # y ← W*x + y

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
function restore!{T}(a::LSTM{T}, filename_W::String, filename_b::String)
    H = length(a.c)
    D = length(a.input)

    vec = open(io->read_binary_vec(io, T), filename_W)
    a.W[:] = convert_to_column_major_array(vec, (D+H,4H))'

    copy!(a.b, open(read_binary_vec, filename_b))

    a
end



# type GRU <: Layer

# end