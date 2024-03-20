# fusedbun

**Note**: This is an attempt to borrow elements from sophisticated optimizers, while being mindful of resource utilization.
Im still not expecting it to perform better than other optimizers. I'm learning on the go and I'll be sharing my learning journey and devlog as I work on building what I'm aiming for. I guess this is just a dumb project to explore optimizers and geek out a bit. Let's see where this rabbit hole leads. ^-^

## /til/devlog

[19-03-2024]
- [x] Todo: Sparse Update Mechanism
- To update parameters when their gradients surpass a specified threshold(similar to eps threshold here!?!?!)

[19-03-2024]
- The conceptual design adapts SM3’s approach to memory efficiency by maintaining compact accumulator data for each parameter(correct me if i'm wrong, but what it does is it tracks the historical magnitude of gradients for each parameter and this minimizes the meomry required to retian historical information).
- Keeping up the good vibes with momentum(it basically allows the optimizer to 'remember' the direction of previous updates) to smoothen the optimization trajectory and gradient centralization(inspired by adalite).
- Lol this is in its awkward phase of development and I tested it against the Algoperf benchmark. FusedBun: 91.66%, AdamW: 88.19% on MNIST. WTFFFF. Try for yourself. 

> .>be me
>
>.>noob
>
>.>decide to spice up my coding game
>
>.>decide to summon GPT-9000-Turbo for some programming magic
>
>.>time to share
>
>.>unoob yourself, whispers the GPT-9000-Turbo

