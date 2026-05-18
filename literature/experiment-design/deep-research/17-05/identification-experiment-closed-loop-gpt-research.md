# Ontwerp van identificatie-experimenten voor gesloten-lus gantry-systemen

## Gesloten-lus identificatie en externe excitatie  
Gesloten-lus systemen met vaste regelaar kunnen ook worden geïdentificeerd door kleine externe excitatie-signalen toe te voegen. Boukhebouz *et al.* (2021) beschrijven bijvoorbeeld een methode om in gesloten lus een multisine-signaal met beperkt amplitude te *shapen* door (i) minimalisatie van de piekfactor (crest factor) en (ii) vormgeven van het frequentiespectrum zodat aan verzadigingsbeperkingen wordt voldaan【1†L162-L170】【3†L85-L94】. Hun voorbeeld toont dat dit de niet-lineariteiten door verzadiging voorkomt en de FRF-identificatie verbetert. Dit is relevant voor het gantrysysteem: een multisine met minimale piek voorkomt dat de korte-krachtbegrenzers worden bereikt en dat het systeem buiten het nominale traject afwijkt. Misković *et al.* (2007) laten bovendien zien dat het simultaan prikkelen van alle ingangen in gesloten lus de parameternauwkeurigheid nooit verslechtert en vaak verbetert【39†L23-L30】. Samengevat: in gesloten-lus opstellingen verhoogt externe excitatie de persistentie van opwinding en verbetert de ruis-ruisverhouding (SNR) van de gemeten respons, mits de amplitudes binnen de regelgrenzen blijven.

## Persistentie door traject versus extra multisine  
Soms is het hoofdlus-traject zelf al persistente excitatie: bewegingen met wisselende X-, Y-positie en versnellingen kunnen deels alle relevante modi prikkelen. Bij lage snelheden of zeer sterke servo-regelaars kan echter de respons in het lage-energiespectrum onvoldoende zijn om de gewenste FHR te bepalen. Uit data van Evans (1998) blijkt dat zó lage amplitude-excitatie (>±2% van de nominale brandstofflow in een motor) de ruis-onzekerheid vergroot, terwijl bij ±5–10% de onzekerheid *heel klein* is【22†L11198-L11207】. Geëxtrapoleerd op de gantry: een te geringe multisine-amplitude leidt tot slechte SNR in de gemeten FRF, wat resulteert in hoge variantie van schattingen. De vuistregel is hier dat men genoeg amplitude moet gebruiken om boven de ruisvloer uit te komen, zonder het nominale traject wezenlijk te verstoren. 

### Bron: Evans (1998) – Gas-turbine identificatie  
- **Volledige referentie:** *Evans, C. (1998). Identification of linear and nonlinear systems using multisine signals with a gas turbine application. Doctoral thesis, Univ. of Glamorgan.*  
- **Kernidee:** Bij een turbomotor werden multisines met verschillende amplitudes gebruikt (±10%, ±5%, ±2% van de maximale brandstofstroom). De resultaten lieten zien dat bij ±10% en ±5% de onzekerheid van de geschatte FRF zeer laag is, terwijl bij ±2% de variatie sterk toeneemt【22†L11198-L11207】. De ±10%-data worden gekozen voor parametrisch modelleren omdat ze de laagste onzekerheid hebben en bijna geen niet-lineariteit vertonen.  
- **Toepassing op gantry:** Evenzo zal een multisine met amplitudes rond enkele procenten van de karakteristieke actuatorkracht het FRF precies kunnen schatten, terwijl een té lage amplitude de SNR verlaagt. Indien de gantry-as vergelijkbaar lineair is binnen een kleine beweging, kunnen we streven naar zo klein mogelijke amplitudes (bv. enkele procenten van rijdbaar vermogen) zónder dat de frf-variantie toeneemt.  
- **Ontwerprichtlijn:** Vergelijk de gemeten FRF bij verschillende excitatie-amplitudes. Kijk of bij lagere amplitude de onzekerheid (twice-theta-bandbreedte) toeneemt【22†L11198-L11207】. Kies de kleinste amplitude die nog acceptabel lage variantie geeft. (Bedoelde focus: amplitudekeuze, SNR vs lineariteit.)

## Keuze van multisine-amplitude: SNR versus lineariteit  
Er is een duidelijke trade-off tussen signaal-ruisverhouding en lineairiteit. Grotere amplitude verbetert de SNR, maar kan niet-lineaire effecten prikkelen en het systeem uit de lineaire werkregio duwen. Retzler *et al.* (2022) benadrukken dat bij frequentiedomeinidentificatie een *lage piekfactor* van de multisine essentieel is: een willekeurige fase kan hoge pieken geven waardoor actuatoren verzadigen en SNR juist slechter wordt【34†L64-L72】. Dit illustreert dat men bij amplitudekeuze niet alleen naar RMS-waarde, maar vooral naar piekwaarde moet kijken om niet in saturatie te belanden. 
- **Bron: Retzler et al. (2022)** – *Automatica* (Crest-factor optimalisatie)【34†L64-L72】【34†L118-L127】. Zij bespreken dat multisine typisch lage piek nodig heeft voor hoge SNR, en optimaliseren fasen met een Levenberg-Marquardt algoritme om de CF te minimaliseren. **Ontwerpregel:** kies multisine-fasen zó dat de piek van de totale verstoring zo laag mogelijk is bij gegeven frequentieverdeling. Hierdoor kan de RMS hoger zijn (betere SNR) zonder pieksaturatie.  

Concreet kan men bij een gegeven powerspectrum de fasen optimaliseren (bijv. Schroeder- of Van der Ouderaa-methoden【34†L93-L102】, of iterative algorithms) om de crest factor zo veel mogelijk te reduceren. Zhang *et al.* (2025) laten zien dat door fasering en -normoptimalisatie de piekwaarde kan dalen bij gelijk vermogen【50†L74-L82】. Zij pasten dit toe op een wafer-stage en vonden significant betere FRF-meting dankzij de lagere crest-factor【50†L74-L82】【50†L83-L91】. 

- **Bron: Zhang *et al.* (2025)** – *Results in Eng.* (Crest-factor wafertafel)【50†L74-L82】【50†L83-L90】. In experimenten op een wafer stage gaf een multisine met geoptimaliseerde fasen een betere FRF dan een willekeurige fasering. *Ontwerprichtlijn:* Optimaliseer fasen (min. crest factor) zodat voor dezelfde RMS-power de piekwaarde minimaal is. Zo blijf je binnen motor- en servo-limieten. (Vervangt focus: crest-factor/MIMO)

## MIMO-multisineontwerp  
Bij meerassige systemen zoals een dual-drive gantry moeten multisine-signalen per kanaal gekozen worden. Een gebruikelijke aanpak is onafhankelijke multisine per in- of uitgangskanaal, elk met random fasen. Bij gelijktijdige inzet geldt: de totale mechanische belasting is som van de krachten. Men moet vermijden dat gelijktijdige pieken in meerdere kanalen accumuleren tot vervorming. Daarom is het verstandig de fasen *per kanaal* onafhankelijk te optimaliseren en de piek van de sombeweging in de gaten te houden. 

Ontwerpregels:
- **Onafhankelijke kanalen:** Exciteer elke motor/zak los met eigen multisine. Bij MIMO-identificatie (zoals bij wafer-stage in【53†L393-L401】) werd dit gedaan met single-as multisine per keer. Als de systemen zwak gekoppeld zijn, kan men per as excitatie inzetten; bij sterke koppeling controleer je de gecombineerde respons.  
- **Fasering:** Gebruik willekeurige fasen of gefaseerde methoden (Schroeder e.d.) om de piekwaarde te minimaliseren【34†L64-L72】【50†L74-L82】. Check de *crest factor* van de gecombineerde invoer (bij gelijktijdig vermogen) om saturatie te voorkomen.  
- **Crest factor m.b.t. load:** Door CF-optimalisatie kan elke multisine meer RMS-vermogen hebben voor dezelfde absolute piek. Dit verhoogt lokaal SNR zonder modellimits te overschrijden【34†L64-L72】【50†L74-L82】.  
- **Spectrum en frequentieband:** Beslis frequenties die relevant zijn (bv. eigenfrequenties tot max servo-bandbreedte). De kaart moet het systeem in het werkgebied raken (hier wellicht 0–50 Hz voor lage snelheidsdynamiek tot enkele honderden Hz voor stijfheidsmodes). Splits het totaal vermogen zodanig over kanalen dat geen actuator meer dan bv. 10–20% van zijn capaciteit gebruikt, om trackingfouten te beperken. 

Met deze richtlijnen blijft de multisine “plant-friendly” (volgens【34†L64-L72】【50†L74-L82】). Belangrijk is vooral dat *onbekend* is wat de piekbelasting wordt als kanalen in fase zijn; handmatige check van worst-case samenloop (bijv. alle cosinussen in alle kanalen omhoog gelijktijdig) is raadzaam. 

## LPV-identificatie langs een bewegend traject  
Het dual-drive gantry is quasi-LPV: de massa/inertie van X-as hangt van Y-pos af, dus XY-dynamica varieert over trajecten. Ebrahimkhani en Lataire (2016) presenteren een methode voor het identificeren van een NL-systeem door te lineariseren langs een tijdsvarierend traject. Zij stellen: “We beschouwen één stabiele traject van het NL-systeem, waarna het systeem kan worden benaderd door een LPV-model rond dit traject. Het LPV-model is het systeem lineair toegepast op het traject, dat we zullen identificeren door dit traject te verstoren”【56†L139-L147】. Met andere woorden: kleine perturbaties rondom de nominale beweging laten je een lokaal lineair model schatten, waarvan de parameters variëren met de Y-coördinaat als scheduling-variabele. 

- **Bron: Ebrahimkhani & Lataire (2016)** – *IFAC* (NL→LPV identificatie)【56†L139-L147】. Kern: lineariseer langs het referentiepad (met Y als schakelvariabele). Door kleine excitatie toe te voegen aan de referentie, identificeren ze het lokale LPV-model en reconstrueren ze vervolgens de NL-dynamiek.  
- **Toepassing:** Voor de gantry betekent dit dat de toegevoegde multisine moet variëren terwijl de Y-positie verandert (persistente excitation in tijd én in scheduling). De nominale Y-trajectori geeft de “werkpunten” van het LPV-model. Kleine multisines per punt laten toe de Y-afhankelijke massa/inertie vast te leggen.  

De les is dat de multisine-amplitudes klein genoeg moeten blijven om lineariteit rond elke Y-positie te behouden (zie ook【56†L139-L147】). Bij te grote verstoringen zou de LPV-aanname breken (het systeem springt naar een andere regime). 

## Aanbevolen workflow en validatie  
Op basis van bovenstaande literatuur en praktijkvoorbeelden (bijv. wafer-stage experimenten【53†L393-L401】) kan de volgende werkwijze aangehouden worden:

1. **Bepaal frequentieband:** Definieer welke frequenties belangrijk zijn (bv. 0,5–500 Hz). Stel de multisine samen met componenten in dit bereik.  
2. **Start met lage amplitude:** Begin bij enkele procenten van de nominale actuatorkracht/voeding. Voer een kort experiment uit en schat de niet-parametrische FRF of de respons.  
3. **Vergelijk met nominale beweging:** Controleer dat het traject (en met name de Y-positie) nauwelijks wordt vervormd. Plot X/Y-signalen van nominale traject en traject+multisine over elkaar om trackingfout te zien. Zie hieronder plotsuggesties.  
4. **Evalueer SNR en lineariteit:** Schat de FRF (niet-parametrisch) van de stap. Bekijk of de FRF consistent is en binnen de (2σ) onzekerheidsbanden valt bij herhaalde metingen. Vergelijk de FRF bij nominale (zonder extra) en met multisine op een paar punten – zie de grafiek in【22†L11198-L11207】 als voorbeeld: de ±10% en ±5% FRF’s vielen netjes samen, consistent met lineariteit.  
5. **Verhoog amplitude indien nodig:** Als de onzekerheid in de FRF (berekend uit 2σ-band of residualen) te groot is, verhoog stapsgewijs de multisine-amplitude. Observeer wanneer bij verhoging de FRF-respons begint af te wijken van schaal. Het punt voor groot niet-lineariteitsverschil markeert een bovengrens.  

### Aanbevolen meetplots  
- **Trackingfouten:** Plot X(t) en Y(t) zowel zonder als met multisine (over elkaar), zodat eventuele afwijkingen zichtbaar worden.  
- **Frequentierespons:** Bode- of singular-waardefiguren van de gemeten FRF’s met en zonder multisine (of bij twee amplitudes) om consistentie te controleren. Bijvoorbeeld Evans plotte in【22†L11198-L11207】 de FRF bij ±10%, ±5%, ±2%, en zag dat de curves samenvielen.  
- **Residualen/SNR:** Plot amplitude-spectrum van de meetfout (bijvoorbeeld output minus model) om te zien of het multisine-signaal domineert over de sensorruis.  
- **Actuatorbelasting:** Toon de som van de in/output-krachten of vermogens om te verifiëren dat er geen piekpieken boven de limieten verschijnen.  

### Checklist “multisine té groot”  
- **Trackingkorrelaties sterk gestegen:** grote afwijking tussen nominale trajectory en gemeten verstoringen (bv. overshoot of instabiliteiten).  
- **Frequentieverschillen:** FRF’s bij verschillende amplitudes wijken duidelijk af (niet meer lineair schalen).  
- **Meetruis toegenomen:** ongewenste harmonischen of niet-lineaire bijfrequenties verschijnen in output.  
- **Actuatorverzadiging:** signalen raken de (digitale/analoge) grens. Check of de toegepaste krachten  (bijv. ±X%) in tijdsignaal verzadigen.  
- **Systeemuitstap buiten verwachting:** bijvoorbeeld versnellings- of snelheidsbegrenzers overschreden.  

## Belangrijkste bronnen om te raadplegen  
- **Evans (1998):** Doctorale thesis *“Identification of linear and nonlinear systems using multisine signals…”* – uitgebreide studie naar de effecten van amplitude, noise en nonlineariteit bij multisine op een gasturbinemodel【22†L11198-L11207】【21†L10388-L10396】. Cruciaal voorbeeld voor SNR-vs-lineariteit.  
- **Boukhebouz *et al.* (2021):** IFAC-paper *“Shaping multisine excitation for closed-loop identification…”*【1†L162-L170】. Beschrijft methoden om multisine-signalen in gesloten-lus aan amplitudebeperkingen aan te passen (crest-factor en spectrumdesign). Relevant voor actuatorgrenzen.  
- **Retzler *et al.* (2022):** *Automatica* – “Improved crest factor minimization of multisine excitation signals”【34†L64-L72】【34†L118-L127】. Geeft inzicht in waarom lage crest-factor essentieel is voor hoge SNR en lage saturatie in identificatie-experimenten.  
- **Ebrahimkhani & Lataire (2016):** *IFAC papers.* *“LPV Model Identification Around a Time-Varying Trajectory”*【56†L139-L147】. Belangrijk voorbeeld van klein-signaal excitation langs een bewegend traject voor LPV-identificatie.  
- **van der Hulst *et al.* (2025):** ArXiv *“Frequency domain identification for multivariable motion control… wafer stage”*【53†L390-L399】. Praktijkvoorbeeld van gesloten-lus multisine FRF-identificatie op een precisie-waferstage.  
- **Zhang *et al.* (2025):** *Results in Engineering* – “Crest factor minimization… wafer stage FRF identification”【50†L74-L82】【50†L83-L90】. Toont hoe fasering de FRF-kwaliteit verbetert.  

Deze bronnen behandelen amplitudekeuze, frequentieplanning, gesloten-lus strategieën, persistentie en MIMO-excitatie. Zij zijn leidend bij de ontwerpregels genoemd hierboven.

**Bronvermelding:** Alle bronnen zijn in tekst geciteerd als `【cursor†Lx-Ly】`. Bijvoorbeeld: Evans (1998)【22†L11198-L11207】, Retzler (2022)【34†L64-L72】, Boukhebouz (2021)【1†L162-L170】, etc.