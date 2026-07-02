---
layout: post
title: "Objects Should Emerge, Not Be Built: Object-Centric World Models from Web Video"
date: 2026-07-02 12:00:00 +0800
categories: [ai, world-models]
---

*A sketch of the representation I think embodied world models want: object-centric, but with objecthood left to emerge from prediction over time rather than baked into the architecture — and just addressable enough to hang physics and dynamics off of.*

## The representation is the whole ballgame

If you want an embodied agent to *act by prediction* — imagine a future, evaluate it, pick the action that gets you there — then the agent needs a model of how the world evolves. That is a world model. And the single most consequential design decision in a world model isn't the architecture of the predictor or the size of the dataset. It's the **latent space you choose to predict in**.

That space determines everything downstream: how expensive training is, what the model is forced to pay attention to, whether you can inject prior knowledge, and whether the learned dynamics are simple enough to actually plan with. Get the representation wrong and you spend your capacity modeling things that don't matter. Get it right and the hard parts — physical consistency, sample-efficient dynamics, compositional generalization — start to come cheaply.

So the question I want to sit with has two halves: *what is the right unit of representation for a world model that learns from web video — and where should that unit come from?*

## The current spectrum: pixels on one end, a global blur on the other

Today's self-supervised video models mostly live at two extremes of a compression axis.

**On the dense end, generative video models** — the video-diffusion lineage (Sora, Cosmos, and friends) — represent a frame as a grid of patch tokens and learn to reconstruct or denoise pixels. This is maximally expressive and produces gorgeous samples, but it's the wrong objective for a *world* model. You burn enormous capacity modeling texture, lighting, and background clutter that have nothing to do with the dynamics you care about. Prediction is expensive, and the representation is entangled: nothing in it says "this is an object and that is another."

**On the compressed end, the JEPA family** — I-JEPA, V-JEPA, and V-JEPA 2 — makes the key move of predicting in *latent space* rather than pixel space. You never reconstruct pixels; you predict the features of masked regions. V-JEPA 2 scaled this to a million-plus hours of internet video and then post-trained an action-conditioned variant (V-JEPA 2-AC) that plans on real robots from a few dozen hours of interaction data. LeJEPA (Balestriero & LeCun, 2025) pushed the *theory* forward, replacing the pile of JEPA heuristics with a single principled objective that drives embeddings toward an isotropic Gaussian. This whole family is dramatically more efficient than pixel prediction, and it clearly works.

But the JEPA representation is still, in a sense, *structureless*. It already predicts over time — that's the whole objective — yet the temporal signal shapes a *distributed* representation and never crystallizes into discrete entities. Whether you keep the full grid of patch latents or pool everything into a compact global summary token, there's no explicit notion of "how many things are in this scene and which is which." The representation is efficient and semantically rich in aggregate, but it doesn't carve the world at its joints.

**And that's the gap.** It's tempting to read it as just a granularity to dial in — something between one-token-per-patch and one-blurry-summary-per-frame. But that undersells it. What both ends actually lack is *structure*: units that line up with the things the world is made of and that you can follow as they change. And structure like that isn't a setting you pick — it has to come from somewhere. The rest of this post argues the somewhere is *time*: the work of predicting how a scene changes is what makes object structure surface in the first place, which answers both halves of the question at once — the right unit is the object, and it should come from prediction rather than architecture.

## The proposal: let objects emerge, then address them

The unit of representation should be the object — but I don't think you should *build* objects into the architecture. The usual tools here, slot attention and object queries, were designed for static images, and they *impose* the decomposition: they decide up front that a frame is `K` competing slots and force the pixels to fight over them. A still frame gives them nothing better to grab onto. On video we can do something more natural: let objecthood *emerge* from the learning problem, and keep only a thin mechanism for *addressing* what emerges.

**Emergence, driven by time.** The oldest cue for what counts as an object is the Gestalt principle of common fate — things that move together belong together. A still image can't offer that signal; video hands it over for free, which means objecthood is actually *easier* to discover in video than in images. And we know a generic objective can find it: DINO showed that a self-supervised ViT with no object machinery at all grows attention maps that segment objects cleanly. So the backbone can stay maximally generic — a spatiotemporal encoder trained to predict future latents (JEPA-style, minimal inductive bias), where the pressure to explain change over time does the grouping. Objecthood falls out because carving the scene into coherently-moving, sparsely-interacting parts is precisely what makes the future cheap to predict.

That last point is worth promoting to a *definition*: an object is the factorization of the state that makes the future easiest to predict — a unit whose dynamics are largely self-contained and interact only sparsely. Defined this way, objecthood is inherently temporal (you can't even state it without the time axis), and it surfaces the *relevant* objects automatically. A shadow or a reflection isn't a mechanism with its own dynamics, so it never earns a factor; a rigid body that moves and collides does. You get dynamically-relevant objecthood instead of exhaustive segmentation — exactly what a world model needs.

**Addressing, with a thin readout.** Emergence alone gives you *implicit* structure: objecthood living in attention patterns, not in variables you can index. But the physics losses and per-object actions below need handles they can grab. So put the object interface on *top* of the generic backbone, not inside it — a light, late-bound query head that reads the emergent structure out into a small set of addressable slot tokens. Crucially, don't fix the count: over-provision queries and let a sparsity pressure retire the empty ones, so the number of objects is data-driven rather than a hyperparameter. Slots are demoted from *how the model perceives* to *how we address what it already learned*.

Why this middle is the right one:

- **Efficiency.** A handful of addressed objects is a far smaller state than a grid of patch tokens, so the transition model that predicts dynamics stays cheap — and the capacity goes toward *things* rather than texture.
- **Semantics for free.** Because the units emerged from predicting motion, each is close to a real entity you can name, track, and reason about — not a patch, not a scene-wide blur.
- **Compositionality.** A state factored into entities generalizes to new arrangements of familiar objects, which a monolithic scene-vector simply can't express.

The bet is that objecthood is close to the natural granularity at which the world is *predictable* — and that the cleanest way to reach it is to let prediction over time discover it, rather than legislate it in advance.

### A concrete instantiation

Vision aside, what would I actually build? Here's the stack I'd bet on, with the reasoning.

**Backbone: a plain joint space-time ViT.** If the philosophy is "impose nothing and let prediction do the work," the honest architecture is the most generic one that scales: patchify the clip into spatiotemporal tubelets (e.g. 2 frames × 16×16 pixels), add 3D rotary position embeddings, and run *full joint attention* over all tokens. Notably, not factorized space-time attention — factorization is itself an inductive bias, a decision that space and time should be processed separately, which is exactly the kind of legislating this proposal argues against. Joint attention lets a token attend to "the same object, three frames ago, twenty patches away" in one hop, and that free-form spatiotemporal routing is the substrate that common-fate grouping actually lives in. The cost is quadratic attention over the clip, but the recipe is proven at web scale (it's essentially what V-JEPA 2 runs); efficiency tricks for longer clips are engineering, not principle. One small, load-bearing addition: **register tokens**. Large ViTs dump global computation into a few high-norm patch tokens, polluting exactly the attention maps the emergence story depends on; a handful of registers absorbs that traffic and keeps the maps clean.

**Objective: block-causal, multi-horizon latent prediction.** Architecture and objective are inseparable here. Masked denoising is fine for representation learning, but a world model needs the temporal arrow: a narrow block-causal predictor (much smaller than the encoder) takes latents up to time $$t$$ and predicts latents at $$t+\Delta$$, with targets from an EMA teacher — or, in keeping with this post's fewer-heuristics spirit, a LeJEPA-style isotropy regularizer in place of the EMA/stop-gradient machinery. Predicting at *multiple* horizons $$\Delta$$ matters for emergence: short horizons reward grouping by instantaneous motion, long horizons reward grouping by persistent identity, and objects are the structure that's stable under both.

**Why emergence happens in this stack.** Under joint attention with a future-prediction loss, the cheapest way to predict a patch's future is to attend to whatever moves coherently with it — its object — because that's where the predictive information concentrates. Gradient pressure therefore organizes attention into motion-coherent groups. DINO's segmentation maps are the static preview of this effect; temporal prediction should sharpen it, since motion disambiguates precisely the cases where appearance alone can't separate figure from ground. The resulting structure is *implicit* — living in attention patterns and feature-space clustering — which is exactly why the readout exists.

**Readout: over-provisioned queries with competitive cross-attention.** Two or three layers of learned queries (over-provisioned, say 32–64) cross-attend into the backbone's spatiotemporal feature map, each carrying a presence logit under sparsity pressure so the effective object count is data-driven. One deliberate bias inside the readout: make the cross-attention *competitive* — normalize over queries at each feature location, slot-attention style — so every patch of the video is explained by roughly one query and the objects come out exclusive rather than overlapping. This looks like re-importing the bias the whole section argues against, but the position is coherent: competition in the *backbone* forces perception through a bottleneck and distorts what gets learned; competition in a *thin readout over an already-generic representation* merely partitions structure that exists. The bias has moved from *how the model sees* to *how we index what it saw*. Finally, condition the queries on time — the same query identity, evaluated at each $$t$$ — so each object's per-timestep state (which the mechanics losses below need for $$\dot{q}$$) comes with temporal correspondence by construction, rather than by matching slots across frames after the fact.

**Roads not taken, briefly.** State-space backbones (Mamba-style) are linear in clip length and tempting for long video, but you lose the attention maps that make emergence *inspectable* — a real cost when the central claim is that structure can be read out of them. Token-merging approaches (ToMe-style) put emergent grouping *inside* the backbone via clustering, arguably with even less bias, but the merge is discrete and awkward to hang losses on. Both are defensible; neither is my pick.

In one line: **joint space-time ViT + registers, trained with block-causal multi-horizon latent prediction, read out by time-conditioned competitive queries with presence gating.**

## Why objects make physics losses natural

Here's the payoff I'm most excited about. Once the readout hands you a set of addressed objects — each a token you can index and decode — **physical priors become expressible as losses**, and they weren't before.

Think about what physical law actually constrains. It constrains *objects*: things persist through time, they don't teleport, they move along smooth trajectories, momentum is roughly conserved through interactions, two solids don't interpenetrate. Every one of these is a statement about entities and their states over time.

Try to write those down on a grid of patch latents and you're stuck — there's no stable referent to attach a conservation law to. Try to write them on a single global token and there's nothing to conserve; the object structure has been averaged away. But on a set of persistent slots, each with a state that evolves frame to frame, these become straightforward auxiliary objectives:

- **Object permanence:** the set of active objects, and each one's identity, should persist across time rather than flicker in and out frame to frame — even under brief occlusion.
- **Trajectory smoothness:** a slot's inferred position/velocity shouldn't jump discontinuously.
- **Interaction plausibility:** when two slots' extents overlap, penalize interpenetration; encourage momentum-consistent exchanges.

None of these require ground-truth labels — they're structural regularizers you can apply self-supervised, on top of the latent-prediction objective. They act as an inductive bias nudging the world model toward the small subset of dynamics that are physically realizable, which is exactly what should make it generalize to unseen physical situations rather than memorizing surface statistics. This is the sense in which the model becomes *physically aligned*: not because we hard-coded a simulator, but because the representation made "obeying physics" a thing you can push on with a gradient.

### Grounding the penalties in analytical mechanics

The examples above are hand-picked heuristics. We can do something more principled: derive the penalty from the equations of motion themselves. Classical mechanics tells us *exactly* which trajectories are admissible — the ones satisfying the Euler–Lagrange or, equivalently, Hamilton's equations — and "how badly a trajectory violates those equations" is a differentiable quantity we can minimize. But those equations are written in generalized coordinates, so first we need coordinates.

**Where the coordinates come from: the readout we already have.** The same thin query head that surfaces emergent objects into addressable slots is where we hang a small MLP — shared across slots, so it stays permutation-equivariant and indifferent to how many objects are in frame — that maps each object token $$s_k$$ to its generalized coordinates $$q_k$$, with a second head for $$\dot{q}_k$$ if you'd rather not finite-difference. Use a continuous parameterization for anything rotational (a 6D rotation representation rather than raw Euler angles) so the derivatives the mechanics losses depend on stay well-behaved. This head is also the gradient path that makes the whole scheme bite: the Euler–Lagrange or Hamiltonian residual backpropagates *through* the MLP into the slot encoder, so demanding physical consistency actively reshapes what the slots encode rather than merely scoring them after the fact.

The subtle payoff is that this MLP doesn't have to recover *true*, metric, real-world coordinates — a relief, since monocular web video hands you no absolute scale to recover them from. Lagrangian mechanics is coordinate-covariant: the Euler–Lagrange equations keep their form under any smooth change of generalized coordinates. So if you *learn* the Lagrangian ($$V$$, and if needed $$M$$) jointly with the readout, the MLP and the energy term are free to settle on *some* self-consistent chart in which the dynamics come out simple — no metric grounding required. That's precisely the failure mode that sinks a differentiable simulator, which needs states in real units, yet costs the soft residual formulation nothing. (The coordinate freedom does need the rest of the objective — the latent-prediction loss and the anti-collapse regularizers — to keep it from being satisfied by trivial constant-coordinate solutions.)

**The Lagrangian route.** With a Lagrangian $$L(q, \dot{q}) = T - V$$, admissible motion satisfies

$$\frac{d}{dt}\!\left(\frac{\partial L}{\partial \dot{q}}\right) - \frac{\partial L}{\partial q} = \tau,$$

which for the usual $$L = \tfrac{1}{2}\,\dot{q}^\top M(q)\,\dot{q} - V(q)$$ expands into the manipulator form $$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + \nabla_q V(q) = \tau$$. Read $$q$$, $$\dot{q}$$, $$\ddot{q}$$ off the predicted slot sequence (finite differences across frames, or predict velocities directly to avoid amplifying noise) and penalize the residual:

$$\mathcal{L}_{\text{EL}} = \big\lVert\, M(q)\ddot{q} + C(q,\dot{q})\dot{q} + \nabla_q V(q) - \tau \,\big\rVert^2 .$$

**The Hamiltonian route.** Equivalently, in terms of generalized momenta $$p = \partial L / \partial \dot{q}$$ and a Hamiltonian $$H(q, p)$$,

$$\dot{q} = \frac{\partial H}{\partial p}, \qquad \dot{p} = -\frac{\partial H}{\partial q} + \tau,$$

giving the residual $$\mathcal{L}_{\text{H}} = \lVert \dot{q} - \partial_p H \rVert^2 + \lVert \dot{p} + \partial_q H - \tau \rVert^2$$. And since an unforced autonomous system conserves its Hamiltonian, you get an energy-conservation penalty for free — $$\mathcal{L}_{\text{energy}} = \lvert dH/dt \rvert^2$$ over any window where nothing is doing work on the object.

Three things make this practical rather than merely elegant:

- **You don't need to know the physics in advance.** For a known system, plug in $$M$$ and $$V$$ directly. For an unknown one, parameterize $$V(q)$$ — and if needed $$M(q)$$, or $$H$$ itself — with small networks. The residual stays a soft penalty, and you've recovered the spirit of Hamiltonian and Lagrangian Neural Networks, where the equations of motion *are* the structure and conservation laws hold by construction. The difference here: we keep an explicit latent-prediction model and add mechanics as a regularizer on top, rather than replacing the learned dynamics with it.
- **The forcing term is your latent action.** Pure Euler–Lagrange and Hamilton assume a closed, conservative system — no friction, no contact, no external push. Real objects get shoved, dragged, and damped. The $$\tau$$ on the right-hand side of every equation above is exactly the generalized force needed to explain the observed motion — which is to say, it *is* the per-object latent action from the next section. The physics residual and the latent-action inference aren't two mechanisms; they're the same equation read two ways. Where you trust an object to be in free flight, set $$\tau = 0$$ and demand strict conservation; where it's interacting, let the inferred latent action absorb the non-conservative forces (a Rayleigh dissipation term or a port-Hamiltonian formulation handles friction more gracefully if you need it).
- **It stays soft.** All of this enters as a weighted auxiliary loss, not a hard constraint. It biases the representation toward dynamically consistent states without forbidding the model from capturing what the idealized equations leave out — precisely the property you want from a regularizer rather than a simulator bolted into the loop.

## Per-object latent actions for efficient dynamics

The last piece is dynamics. Web video has no action labels — nobody annotated what caused each transition. The trick, pioneered in the latent-action line of work (Genie's latent action model; LAPO-style latent-action-from-observation), is to *infer* a latent action that explains the transition from frame `t` to frame `t+1`, and train a forward model conditioned on it. This is also, essentially, what V-JEPA 2-AC does once you bolt a real action space on afterward.

The extension I want to make is: **factor the latent action per object.** Instead of one global latent action explaining the whole scene's change, infer a small latent action (or transition code) *per slot* — the thing that transforms each object's state from one step to the next.

Two reasons this should help:

1. **Sample efficiency.** Most objects in most frames are doing something simple: staying still, translating, being acted on by one neighbor. A per-object action space lets the model reuse "an object is pushed left" across every object and every scene it ever appears in, instead of relearning it inside a monolithic scene transition. The dynamics become compositional in the same way the state is.
2. **Controllability and counterfactuals.** Per-object latents give you a natural interface for intervention: hold every object's action fixed and vary one, and you get a clean "what if this thing moved differently" rollout. That's precisely the query an embodied planner wants to make.

Put crudely, the state is factored into objects, so the dynamics should be factored into per-object transitions — and the whole system becomes a set of interacting entities whose pairwise effects the model can learn, rather than one giant undifferentiated next-state function.

## A training recipe, in one breath

Stitching the pieces together, the self-supervised loop looks roughly like:

1. Encode the clip with a generic spatiotemporal backbone (no object machinery), then read its emergent structure out into a small, data-driven set of addressable object slots via a thin late query head.
2. Infer a per-slot latent action explaining each frame-to-frame transition.
3. Roll the slots forward with a transition model conditioned on those latent actions.
4. Train the prediction in **latent space**, JEPA-style — predict future slot features, not pixels. (This objective is also what pressures objecthood to emerge in the backbone in the first place.)
5. Add the physics regularizers (permanence, smoothness, non-interpenetration) as auxiliary losses.
6. Later, ground the latent actions in a real action space with a small amount of embodied interaction data, exactly the V-JEPA-2-AC recipe, and plan.

Everything up to step 6 runs on unlabeled web video.

## The honest part: what makes this hard

I don't want to oversell it. The reason object-centric world models aren't already the default is that getting clean, usable objecthood out of real video is genuinely unsolved at scale:

- **Does objecthood actually emerge cleanly?** DINO-style emergence is real but coarse. Getting a predictive backbone to factor cluttered, real-world video into *separable* objects — rather than a smear of half-bound parts — is still fragile off synthetic benchmarks, and it's the load-bearing assumption of the whole approach.
- **Temporal correspondence.** Keeping an object's identity stable across occlusion, entry, and exit is its own research problem, readout or not.
- **Does the sparsity prior find the right granularity?** Letting the object count be data-driven is nicer than fixing `K`, but "the right number of objects" is scene-dependent, and the prior can happily under- or over-segment.
- **Evaluation.** "Did the world model learn good physics?" is much harder to score than "did the pixels look right," and we lack great benchmarks for it.

But none of these feel fatal, and the upside — an efficient, interpretable, physically-grounded, compositionally-controllable world model — is worth the fight. The generative-video and JEPA camps have each proven half the point: that you can learn from web-scale video, and that you should predict in latent space. The missing ingredient is *structure*. My bet is that objects are that structure.

---

### Further reading

- V-JEPA 2: *Self-Supervised Video Models Enable Understanding, Prediction and Planning* — Assran et al., 2025 (arXiv:2506.09985)
- LeJEPA: *Provable and Scalable Self-Supervised Learning Without the Heuristics* — Balestriero & LeCun, 2025 (arXiv:2511.08544)
- *Object-Centric Learning with Slot Attention* — Locatello et al., 2020
- *Emerging Properties in Self-Supervised Vision Transformers* (DINO) — Caron et al., 2021
- *Vision Transformers Need Registers* — Darcet et al., 2024
- DETR: *End-to-End Object Detection with Transformers* — Carion et al., 2020
- SAVi: *Conditional Object-Centric Learning from Video* — Kipf et al., 2022
- Genie: *Generative Interactive Environments* — Bruce et al., 2024
- LAPO: *Learning to Act without Actions* (latent action policies from observation) — Schmidt & Jiang, 2024
- *Hamiltonian Neural Networks* — Greydanus, Dzamba & Yosinski, 2019
- *Lagrangian Neural Networks* — Cranmer et al., 2020
