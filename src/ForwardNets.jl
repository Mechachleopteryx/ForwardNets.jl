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
    GRU,
    Conv2d,
    BatchNorm1,
    BatchNorm3,
    Concatenator,
    Reshaper,
    ReLU,
    ELU,
    TanH,
    SoftPlus,

    ForwardPass,

    read_binary_vec,
    infer_shape,
    convert_to_column_major_array,
    convert_to_column_major_array!,

    forward!,
    add_node!,
    restore!,
    indexof,
    lastindex,
    name,
    output,

    sigmoid,
    relu,
    elu,
    softplus,

    zero!,  # to zero out the hidden state of an LSTM

    calc_forwardpass,
    print_forward_pass

pkgdir = dirname(@__FILE__)

include(joinpath(pkgdir, "read_binary_vec.jl"))
include(joinpath(pkgdir, "utils.jl"))
include(joinpath(pkgdir, "node.jl"))
include(joinpath(pkgdir, "forwardnet.jl"))
include(joinpath(pkgdir, "variable.jl"))
include(joinpath(pkgdir, "layers.jl"))
include(joinpath(pkgdir, "recurrent.jl"))
include(joinpath(pkgdir, "conv.jl"))
include(joinpath(pkgdir, "batchnorm.jl"))
include(joinpath(pkgdir, "activations.jl"))
include(joinpath(pkgdir, "forwardpass.jl"))

include(joinpath(pkgdir, "deprecated.jl"))

end # module