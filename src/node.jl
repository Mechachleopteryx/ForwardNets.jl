abstract Node{T<:Real}

abstract Layer{T} <: Node{T}
abstract Activation{T} <: Node{T}

typealias NameOrIndex Union{Symbol,Int}

Base.eltype{T}(::Node{T}) = T

"""
    name(n::Node)
Return the Symbol name of this Node.
"""
@required_func name(n::Node)

"""
    output(n::Node)
Get the tensor output of the Node
"""
@required_func output(n::Node)


"""
    forward!(n::Node)
Compute the node output
"""
@required_func forward!(n::Node)
