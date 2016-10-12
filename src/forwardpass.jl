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
        print(io, "ForwardPass: ", name(net[forwardpass.activation_order[1]]))
        for i in 2 : length(forwardpass.activation_order)
            print(io, " → ", name(net[forwardpass.activation_order[i]]))
        end
    end
end
Base.show(forwardpass::ForwardPass) = print_forward_pass(STDOUT, forwardpass)

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
calc_forwardpass(net::ForwardPass, input::Symbol, output::Array{Symbol}) = calc_forwardpass(net, [input], output)
calc_forwardpass(net::ForwardPass, input::Array{Symbol}, output::Symbol) = calc_forwardpass(net, input, [output])
calc_forwardpass(net::ForwardPass, input::Symbol,        output::Symbol) = calc_forwardpass(net, [input], [output])

function forward!(forwardpass::ForwardPass)
    for index in forwardpass.activation_order
        forward!(forwardpass.net.nodes[index])
    end
    forwardpass
end
