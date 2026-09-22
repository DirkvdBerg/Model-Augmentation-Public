**ASMPT relevant motion profiles**

trajectories ASMPT gebruikt

setpoint generator op de telica, setpoint generator is van etel zelf. jasper heft die nagemaakt.

trajectories genereren die jasper heft nagemaakt



setpoint van ILC kunnen gebruiken

wat bewegingen die vaak worden gedaan door de machine

long stroke

maandag met jasper en dragan het beste bespreken

welke precies gebruiken



om te starten trajectories van ILC kunnen kijken



minimal beweging die we willen doen. minimale en maximale distance. en acceleraties die we gebruiken. standaarden voor die nu gebruiken in de machines.





Validatie moet eigenlijk op relevante trajectories worden gedaan





Valideren met trajectories

andere controller gebruiken

trainen met bepaalde controller, controller aanpassen en dan kijken of het nog steeds goed klopt bijv.

welke frequenties je aanslaat wordt wel bepaald/afhankelijk van de controller

\--> voordat je een controller op de machine test, kun je eerst op dit model test. usecase controller aanpassen. goede realitische usecase voor hoe ASMPT het model zou gebruiken.



**Presenting data**

in een zin meerdere data, zeggen met meerdere seeds. even benoemen in verslag.

goed zijn om te doen.

vooral om te kijken om tot de beste performance te komen, kan ook unlucky zijn

nf relateren aan langste tijd dynamica dat we nodig hebben. als je kan beargumenteren of als hyperparameter is ook goed. als je het kan beargumenteren en vast zet, goed genoeg/goede weg.



validation settling kan controlleren --> bedoelt evalueren.

plus 300 of 400 ms settling tijd zou een goede

minimal 200 ms tussen zitten. van de ene beweging naar de andere beweging. --> omdat het gedrag tijdens de settling heel belangrijk is.

tijdens validatie settling gedrag evalueren.



Alleen eindresultaat. parameter recovery tussenstap, niet explicite in het verslag zetten.

alleen met eindresultaat werken



parameter recovery dataset voor initial estimate





Plaatje inzoomt op settling. mooi Plaatje tijdens tracking wat het verschil is.

Voor het model vooral het gedrag van hoe die settled, system afhankelijk niet het model.

kijken naar het gedrag voor settelen





Extrapolatie

hogere maximale waarde is ook al

controller aanpassen is ook al extrapolatie





Genoeg data

training en validatie zelfde performance

als performance goed is weet je dat je genoeg data hebt gemaakt



**Thesis outline**

Bij elke beslissing die je neemt uitleggen waarom je die beslissing neemt





**Physical baseline**

maakt niet uit hoe parameters identificeren, jacobian kijken hoe?



**Orthogonality**





limitatie van bepaalde methoden laten zien zoals orthogonality Plaatje. met uniqueness. kan ook in volgende stappen beschrijven.



**Closed-loop**

machines amspt kan ook niet in closed-loop, ook wel een punt.

open-loop trainen en meten niet het problem.

omdat ik in closed-loop moet meten --> relevant voor system, kwam dit problem naar voren





quinten zegt kijkt naar wat er gebeurt met noise in de closed-loop

realistisch mogelijk maken met hoe het in het echt system ook is.

Hoeveel noise voeg je toe: vragen Jasper hoeveel noise er op zit. Kijken naar ILC data om SNR te bepalen. Op basis daarvan een schatting kunnen maken van de noise.



**Data and setup**

bfr of rms een kiezen



black box vrijheid geven om zo groot te worden als die zelf wil. Quinten denkt dat dat wel mag

deepsi maker subnet. kijk naar hoe hij het implementeert.



**Results**

model dichtbij blackbox of zelfs iets beter. Hoeft heel veel zelf niet te leren.
Als we dan gaan extrapoleren, dat black box model faalt maar het augmentation model dan nog wel stand houdt, het liefst beter dan het LTI model.

Twee baselines het beste van beiden wereld.



Beter vergelijken dan vooraf zeggen hoe accuraat je wil. lasting om een requirement geven, dan beter om black box en lti vergelijken en dan interpolatie en extrapolatie en op basis daarvan beslissen of we te vreden zijn of niet.





beginnen met de basis meer toe voegen als je tijd hebt.





**Discussion**

basis functions enzo in intro als je het wil benoemen.

orthogonality, parameter recovery and augmentation.



**Conclusion**

Kijken IEEE standard format om te kijken of het standard is om je research question in je intro te zetten, als dat niet standard is misschien raar om in je conclusive te beantwoorden.





vragen dragan en jasper

inleveren 11 inleveren

