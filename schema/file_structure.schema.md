# JSON Schema

## Properties

- **`version`** *(string, required)*: Version of input file.
- **`fluid`** *(object, required)*: Object used to define parameters related to the circulation fluid.
- **`grout`** *(object, required)*: Object used to define parameters related to grout material.
- **`soil`** *(object, required)*: Object used to define parameters related to soil material.
- **`pipe`** *(object, required)*: Object used to define parameters related the pipes.
- **`borehole`** *(object, required)*: Object used to define parameters related the borehole.
- **`simulation`** *(object, required)*: Object used to define parameters related to simulation.
- **`geometric_constraints`** *(object, required)*: Object used to define geometric constraint parameters for the selected design algorithm.
- **`design`** *(object, required)*: Object used to define parameters related to design and sizing.
- **`loads`** *(object, required)*: Object used to define loads for design and sizing.
