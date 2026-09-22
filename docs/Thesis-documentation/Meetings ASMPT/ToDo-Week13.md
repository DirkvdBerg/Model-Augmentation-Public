**Main to do**

**fix sampling rate / diagnostics and replace with segment\_diag.py, also want in \_\_main\_\_ that everything is plotted/I can run it as a standalone file aswell.**

**check multisine design**

**fix random trajectories always choose from each trajectory**

**fix with potential overlap or mini batches but not the random currently. Fix this in precompute, we can have fixed segments**

**fix precompute and calling different trajectories folder and make sure that its properly saved in the metadata/that its clear what is stored. the seed/version number = 1, is not the best**





**normalization toggle per trajectory or one global (for the channels X1, X2, Y)**

**grouping removed**



**summed parameters need to look at the sum for parameter recovery and can print the individual once**

**what did i add for identifiability**









**kijken om semgent\_diag.py te vervangen voor de diagnostics voor sampling rate en frequency waarvan Maarten sprak**

**Ook file anders noemen**

**Should maybe also plot, FFT or other things not sure how called**

**that i can also run as a standalone file, and in main (only when i run the file) i plots all the stats and necessary figures I want and prints everything.**



Hoi Jasper,

Ik had een vraag over het toevoegen van de additional state. Heeft het vanuit ASMPT meerwaarde als de additional state een fysische betekenis heeft, puur voor simulatie?

Of is het voldoende als ik het alleen als een "proof of concept" heb voor het recoveren van deze toegevoegde state?





ff mailen woensdag 16 mei?





currently no train validation test split

Maarten was not fan of the full trajectory evaluation for scheduling

but why is adam never scheduling its learning rate, also not converged currently



how are the segments calculated in parallel, is this even calculated in parallel?





fix segment lengths

fix trajectories to Garcia

add multisine in feedforward based on lecture



kijken hoe nuttig García's experiments zijn

denk niet dat het alles tegelijk excite?

hier multisine aan toevoegen?

hoe te doen voor train, validation en test. hoe met learning rate schedulen? full trajec evaluation not right. hoe met adam dan? of met validation set?

hoe train val test split, hoe kan ik unseen trajectories overhouden?



**Code volledig begrijpen!!**

**scheduling/validation should be made a toggle for now and turned off. or I should remove it completely because with held out trajectory i will also take sub samples and not full trajectory...**





**Will my additional state be able to be recovered as an additional mass? Jan did the same, so..**

**currently my Centre of Gravity / total mass, or intertia is not correct because of the additional mass**





**Still don't understand the train point Quinten was trying to make**





**look at García experiments, Roland said I think also to use all excited / coupled / all channels active for our gradient approach. If I have all the García experiments am I not underrepresenting this part, should I add more without the grouping?**

**What about the coulomb not currently taken into account?**



**Kijken chat die system-identification lectures heeft gelezen**





**Experiment design:**

**Can add jacobian or fisher information matrix to determine most information gain for each trajectory, but can leave this for experiment design i think**

&#x20;**The only genuine additions worth considering are:**

&#x20; **1. Add d as a trainable parameter — T5 already provides the right excitation for it**

&#x20; **2. Add multisines per trajectory — T4 gets resonance-targeted, others get broadband**

&#x20; **3. Validate with Jacobian/Fisher analysis — confirm the full set is sufficient before committing**

&#x20;  **to hardware experiments**



**Currently still no validation or test trajectory... mis ook Jan vragen voorstellen, dan scheduler..?**



**Is this why learning rate never gets updated inside adam?:     optimizer = torch.optim.Adam(block.parameters(), lr=lr), I always state lr as lr, but adam uses momentum not learning rate updating..?**





**No random stratisfied sampling anymore, just random...**

**Removed groupings?**



**can use overlap over the different trajectories for the segments, but Quinten recommends to use the full trajectory for all 6 for each epoch, or I can use mini-batches, but that differs from how i currently do i with the validate trajectory and random segments each epoch. i can just precompute the overlap between segments of the full trajectory overlap is not required, but possible. And then just use mini-batches for this.**

**remove validate, make predetermined (mini-batches)**



**System identification for closed-loop? our system is still linear right? because the only nonlinearirty for theta we simplified it to be linear**





**Need to add noise!! and methods to work around the noisy data, and realistic amount of noise.**





**Voor validation ook de multisine injected gebruiken? sws kijken validation en test trajectories, hoe bepaal ik dat/kies ik die??**



**gradient windowing also creates observability problem??**





**do my diagnostics take into account how my multisine is generated, and does that matter??**







**should minibatches be length differ per trajectory, or all the same sample length? how much does this matter for "fairness" / information density??**





**noise, validation and test set?? en minibatches fixes, gebasseerd op gevonden segment length, automatiseren.**

**how does Jan work around noise with batch for gradient descent?
and when adding noise do i need Jan's subspace encoder?**





**will this cause an issue:
optimizer = torch.optim.Adam(block.parameters(), lr=lr)**

&#x20;   **scheduler = torch.optim.lr\_scheduler.ReduceLROnPlateau(**

&#x20;       **optimizer, patience=7, factor=0.5, min\_lr=1e-5,**

&#x20;   **)**

