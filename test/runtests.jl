using Base.Test
using ForwardNets


net = ForwardNet{Float32}()
push!(net, Variable(:A, Float32[0]))
push!(net, Variable(:B, Float32, 2))
push!(net, Variable, :C, Float32[1])
push!(net, Variable, :D, 3)
@test lastindex(net) == 4

A = net[:A]
B = net[2]

@test name(A) == :A
@test name(B) == :B
@test output(A) == [0.0]
@test size(output(B)) == (2,)

push!(net, Affine, :affine, :A, 1)
push!(net, Affine, :A, 1)
affine = net[:affine]
@test name(affine) == :affine
@test size(output(affine)) == (1,)

A.tensor[1] = 1.0f0
@test affine.input[1] == 1.0f0
affine.W[1] = 2.0f0
affine.b[1] = 3.0f0
affine.output[1] = NaN
forward!(affine)
@test affine.output[1] == 5.0f0

push!(net, Concatenator, :concat, [:A, :C])
concat = net[:concat]
@test size(output(concat)) == (2,)
output(A)[1] = 1.0f0
output(net[:C])[1] = 2.0f0
forward!(concat)
@test concat.output == [1.0f0, 2.0f0]

push!(net, Reshaper, :reshaper, :concat, (1,2))
reshaper = net[:reshaper]
@test size(output(reshaper)) == (1,2)
forward!(reshaper)
@test output(reshaper) == [1.0f0, 2.0f0]'

push!(net, TanH, :tanh, :A)
push!(net, ReLU, :relu, :A)
push!(net, SoftPlus, :softplus, :A)
_tanh = net[:tanh]
_relu = net[:relu]
_softplus = net[:softplus]

output(A)[1] = -2.0f0
forward!(_tanh)
@test isapprox(output(_tanh)[1], tanh(-2.0))

output(A)[1] = -2.0f0
forward!(_relu)
@test isapprox(output(_relu)[1], relu(-2.0))

output(A)[1] = -2.0f0
forward!(_softplus)
@test isapprox(output(_softplus)[1], softplus(-2.0))

push!(net, LSTM, :lstm, :D, 3)
lstm = net[:lstm]
@test name(lstm) == :lstm
lstm.state[1] = NaN
zero!(lstm)
@test lstm.state[1] == 0.0f0
forward!(lstm)

push!(net, Variable, :output, :affine)

pass = calc_forwardpass(net, [:A], [:output])
output(A)[1] = 2.0f0
forward!(pass)
@test output(net[:output])[1] == convert(Float32, 2*2 + 3)