#!/usr/bin/env julia

using Yaqs

const MPOMod = Yaqs.MPOModule
const OUTPUT = joinpath(@__DIR__, "julia_hs_mpo_nonzero.csv")

mpo = MPOMod.init_haldane_shastry(16; J=1.0, pbc=true)
open(OUTPUT, "w") do io
    println(io, "site,left,phys_out,phys_in,right,real,imag")
    for (site, tensor) in enumerate(mpo.tensors)
        for index in CartesianIndices(tensor)
            value = tensor[index]
            if !iszero(value)
                left, physical_out, physical_in, right = Tuple(index)
                println(io, join((site, left, physical_out, physical_in, right, real(value), imag(value)), ','))
            end
        end
    end
end
