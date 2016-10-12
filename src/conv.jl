type Conv2d <: Layer
    name::Symbol
    W::Array{Float32, 4} # filter_h × filter_w × in_c × out_c
    b::Vector{Float32} # out_c
    strides::Tuple{Int, Int} # width, height

    parent::Array{Float32, 3} # h × w × in_c
    child::Array{Float32, 3}  # out_h × out_w × out_c
end
get_name(a::Conv2d) = a.name
output(a::Conv2d) = a.child
function add_node!(net::ForwardNet, ::Type{Conv2d},
    name::Symbol,
    parent_index::Int,
    strides::Tuple{Int, Int},
    filter_h::Int,
    filter_w::Int,
    out_c::Int)

    parent = output(net[parent_index])::Array{Float32, 3}
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
function restore!(a::Conv2d, filename_W::String, filename_b::String)

    filter_h, filter_w, in_c, out_c = size(a.W)

    vec = open(read_binary_vec, filename_W)
    a.W[:] = convert_to_column_major_array(vec, (filter_h, filter_w, in_c, out_c))

    copy!(a.b, open(read_binary_vec, filename_b))

    a
end