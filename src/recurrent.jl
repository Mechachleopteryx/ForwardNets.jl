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

#=
u = f(W_xu * x + W_hu*h_{t-1} + b_u)
r = f(W_xr * x + W_hr*h_{t-1} + b_r)
c = tanh(W_xc * x + (W_hc*h_{t-1}) .* r + b_c)
h_t = (1-u) .* h_{t-1} + u .* c
=#
type GRU{T} <: Layer{T}
    name::Symbol
    W_x::Matrix{T} # 3H×D
    W_h::Matrix{T} # 3H×H
    b::Vector{T} # 3H

    input::Vector{T} # input, [D]

    # preallocated memory
    xb_ruc::Vector{T}  # [3H]
    xb_r::Vector{T} # [H]
    xb_u::Vector{T} # [H]
    xb_c::Vector{T} # [H]

    h_ruc::Vector{T} # [3H]
    h_r::Vector{T} # [H]
    h_u::Vector{T} # [H]
    h_c::Vector{T} # [H]

    r::Vector{T} # [H] input gate
    u::Vector{T} # [H] new input
    c::Vector{T} # [H] forget gate
    h::Vector{T} # [H] GRU state
    h_prev::Vector{T} # [H] previous state
end
name(a::GRU) = a.name
output{T}(a::GRU{T}) = a.h
zero!{T}(a::GRU{T}) = fill!(a.h_prev, zero(T))
function Base.push!{T}(net::ForwardNet{T}, ::Type{GRU},
    name::Symbol,
    parent_index::NameOrIndex,
    H::Int; # hidden layer size
    )

    input = output(net[parent_index])::Vector{T}
    D = length(input)

    W_x = Array(T, 3H, D)
    W_h = Array(T, 3H, H)
    b = Array(T, 3H)

    xb_ruc = Array(T, 3H)
    xb_r = Array(T, H)
    xb_u = Array(T, H)
    xb_c = Array(T, H)

    h_ruc = Array(T, 3H)
    h_r = Array(T, H)
    h_u = Array(T, H)
    h_c = Array(T, H)

    r = Array(T, H)
    u = Array(T, H)
    c = Array(T, H)
    h = Array(T, H)
    h_prev = Array(T, H)

    node = GRU(name, W_x, W_h, b, input, xb_ruc, xb_r, xb_u, xb_c, h_ruc, h_r, h_u, h_c, r, u, c, h, h_prev)
    push!(net, node, parent_index)
end

function forward!{T}(a::GRU{T})

    H = length(a.h_prev)
    D = length(a.input)

    copy!(a.xb_ruc, a.b) # y ← b
    Base.LinAlg.BLAS.gemv!('N', one(T), a.W_x, a.input, one(T), a.xb_ruc) # y ← W*x + y

    copy!(a.h_ruc, Base.LinAlg.BLAS.gemv('N', one(T), a.W_h, a.h_prev)) # y ← W*x

    copy!(a.xb_r, 1, a.xb_ruc,    1, H)
    copy!(a.xb_u, 1, a.xb_ruc,  H+1, H)
    copy!(a.xb_c, 1, a.xb_ruc, 2H+1, H)

    copy!(a.h_r, 1, a.h_ruc,    1, H)
    copy!(a.h_u, 1, a.h_ruc,  H+1, H)
    copy!(a.h_c, 1, a.h_ruc, 2H+1, H)

    # compute u, r, c, and h
    for k in 1 : H
        a.r[k] = sigmoid(a.xb_r[k] + a.h_r[k])
        a.u[k] = sigmoid(a.xb_u[k] + a.h_u[k]) 
        a.c[k] = tanh(a.xb_c[k] + a.r[k]*a.h_c[k]) 
        h_old = a.h[k]
        a.h[k] = (1 - a.u[k])*a.h_prev[k] + a.u[k]*a.c[k]
        a.h_prev[k] = h_old
    end

    a
end
function restore!{T}(a::GRU{T}, filename_Wx::String, filename_Wh::String, filename_b::String)
    H = length(a.h_prev)
    D = length(a.input)

    vec = open(io->read_binary_vec(io, T), filename_Wx)
    a.W_x[:] = convert_to_column_major_array(vec, (D,3H))'

    vec = open(io->read_binary_vec(io, T), filename_Wh)
    a.W_h[:] = convert_to_column_major_array(vec, (H,3H))'


    copy!(a.b, open(read_binary_vec, filename_b))

    a
end