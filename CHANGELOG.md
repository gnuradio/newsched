# Changelog
All notable changes to newsched will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)

## [0.1.1] - [2021.11.11]

Didn't take long to require a patch from the first release

### runtime
* Propagate dependencies through meson
  * not having this was causing build issues on Fedora

### blocklib
* Adds some missing include files

## [0.1.0] - [2021.11.11]

Here it is: the first release of newsched!  

Newsched is the proof of concept framework for a future GNU Radio 4.0

By releasing newsched in a slightly formal way, the hope is that more developers will 
have access to this framework and learn the concepts that will eventually
find their way into the GNU Radio codebase.  

Development on newsched has been ongoing for over a year, so the codebase
has evolved rapidly in that time - thus there are no details for this first
changelist.  Just consider this the first drop.

### Core Features
- Modular Scheduler Framework
  - interfaces based on a single input queue
  - default scheduler with N blocks/thread
- Custom Buffers
- YAML-driven Block Workflow
- Consolidated Parameter Access Mechanisms
- Simplified Block APIs

Detailed documentation can be found [here](https://gnuradio.github.io/newsched)

With this release of newsched, you can easily create your own blocks, custom
buffers, and even your own scheduler if you are so inclined

Special thanks to Bastian Bloessl and Marcus Müller for leading the effort 
to architect the runtime and provide guidance as to the design decisions

Also want to acknowledge the Scheduler Working Group who have consulted and provided
feedback and ideas on a regular basis about design decisions.  I apologize
if I have left anyone out here, but another special thanks to: Seth Hitefield,
Jeff Long, David Sorber, Mike Piscopo, Jacob Gilbert, Marc Lichtman, Philip Balister,
Jim Kulp, Wylie Standage, Garrett Vanhoy, John Sallay, and all the people associated with 
with the DARPA DSSoC program that shared their research giving valuable insight.

There is much work left to do, so please reach out on chat.gnuradio.org #scheduler
room if you would like to get involved