# Burning fuel for cheap! Transport-independent depletion in OpenMC

## Abstract
SaltProc is an open source tool for simulating batch-wise reprocessing of fuel
in nuclear reactors. SaltProc was initally developed as a wrapper around a
closed source export controlled depletion solver. This design decision limited
the size of the userbase. To try to grow the userbase, I began to contribute
features towards implementing an interface for an open-source alternative, and
eventually became the maintainer of the software, driving design and development
practices for the entire codebase. My poster will focus on how I approached sustainbility
in SaltProc, and specificaly how I used abstract classes to bring modularity to the code,
how using input schema and documentation incereases ease of use for future users, and how
I used GitHub Discussions to foster community with a forum for users to interact with
developers.

## Description
Neutron radiation in the fuel inside of a nuclear reactor induces nuclear
fission in certain nuclides like U235. The fission reaction causes the
nucleus to break apart, releasing both energy and new nuclides, many of which
are radioactive isotopes of smaller elements. This process
is referred to as depletion or burnup. Some of these nuclides inhibit the
nuclear reaction, and greater fuel efficiency can be achieved by chemical
reprocessing of the fuel to remove them.

SaltProc is a code for simulating chemical reprocessing of nuclear reactor fuel
in molten salt reactors. It was originally developed to wrap around Serpent2,
a closed source export controlled Monte Carlo neutron transport code with a depletion solver.
While the original developer of SaltProc had intended for it to be code-agnostic
with respect to the depletion solver used, the small user base meant any contributions
were typically only performed by the core maintainer, so no additional interfaces were
ever developed. Small projects like this typically die out once the author leaves
due to code neglect and lack of community momentum. When OpenMC, a community
developed open source neutron transport code, gained a depletion solver, I wanted
to implement an interface for it to make SaltProc more accessible to new users and
grow the community.

In this poster, I will cover:
- Strategies to increase code sustainability in an open-source code with a small user base
- What could have made the process from going from a user to a developer more smooth
- Utility of technologies like input schemas and CI for code sustainability
- Good approaches for bringing large features into previously existing projects
- Various snags I ran into related to dependencies and how I approached solving them
- The results of my code additions compared to the old implementation.

The intended audience of this poster are folks who are interested in software
sustainability who want learn about what it looks like in a code with a small or
growing community. The intended takeaways are that developing features and maintaining
a code with a small community requires lots of consistency, patience, and planning to
do sustainably.

## Notes
This could also be in the Maintainers and community track.
Link to docpages

Notes from writing session:
- try to focus on things that not everyone might be an expert in.
- Maybe say who your audience is
- Address who the audience for your talk is and what they are going to learn in the talk
- Tell story about how I made the project sustainable (adding documentation, updating variable names and documentation)
- In sustainable, we mean the torch can be passed (e.g. documentation, CI, examples)
- What could have made the process from going from a user to a developer more smooth? Maybe try to frame it this way
- S

