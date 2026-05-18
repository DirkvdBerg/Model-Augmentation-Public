# Ontwerp van stap-/transiënt-experimenten  

We starten met eenvoudige stap- en clutchexperimenten om de traagste plantdynamiek te onderzoeken. Evenwichtige stapwisselingen in één as geven snel inzicht: hoewel het volgen van complexe trajecten het uiteindelijke doel is, toont een gestopte stap hoe de regelaar en plant reageren. Weiss et al. (2020) deden dit voor een printer-gantry: “de staprespons geeft inzicht in het effect van de regelaar op de systeemperformance, ook al is staprespons niet het uiteindelijke doel”【70†L1272-L1275】. Zij benadrukken dat in praktijk stappen onder invloed van snelheids‐/versnellingslimieten gebeuren (bijvoorbeeld de motor bereikt maximaal toerental), zodat de stap niet echt instantaan is. In hun tests resulteert een “stap” in een zogenaamde maximum-velocity ramp【70†L1277-L1285】. Dit betekent dat men stap commando’s moet limiteren (bijvoorbeeld trapezoïdale snelheidsprofielen) om krachten, jerk en doorbuiging binnen veilige grenzen te houden. Stapsignalen kunnen per as afzonderlijk worden toegepast: een stap in X1+X2 tegelijk (common X) versus tegengesteld (differentiële stap voor yaw), en een stap in Y. Zo worden X-vertaling, yaw en Y afzonderlijk geïdentificeerd.  

【91†embed_image】 *Figuur: Staprespons van een gekoppeld pan/gantry-systeem (stippellijn: pan-as, doorgetrokken: gantry-as)【117†L22-L29】.*  

Weiss et al. toonden ook nonlineaire effecten: kleine X-stappen (1 mm) gaven ~50% overshoot, terwijl grote stappen (54 mm) vrijwel geen overshoot veroorzaakten【124†L1298-L1304】. Dit duidt op verzadiging en wrijvingsinvloeden. Onder PI-regeling nam bij hen de overshoot juist toe en werd de demping lager (de PI-regelaar was voor tracking geoptimaliseerd, niet voor staprespons)【124†L1305-L1313】. De (discrete) integrator in de regelaar elimineerde uiteindelijk de fout (na ~0.4 s)【124†L1309-L1313】. Dat illustreert dat een blijvende versnelling in de staprespons in feite een regelsysteem-integrator is, geen planttraagheid. 

Voor ons gantrysysteem raden we vergelijkbare opzet aan: blokkeer de ene as (clutchen of remmen) om de andere as te testen. García-Herreros et al. (2013) deden dit systematisch: eerst blokkeren zij X1 en X2 (X1=X2 constant) en bewegen het Y-vlak met constante snelheid/acceleratie om Y-massa en wrijving te bepalen【91†L677-L686】. Vervolgens bewegen ze X1=X2 synchroon (common mode) om gezamenlijke massa en wrijving van X-as te meten【91†L687-L694】. Voor yaw hingen ze het kruisstuk vast op kleine bekende draaihoeken (±0,1 rad) en bepaalden stijfheid/inertie uit de gemeten krachten【91†L683-L692】. Zij noteren ook de actuatorlimieten (maximale snelheid 2 m/s, acceleratie 20 m/s², yawhoek ±0,1 rad) om tests veilig te houden【91†L663-L671】.   

# Dominante dynamica bepalen  

Uit een staprespons kunnen de traagste tijdsconstante(s) en eigenfrequentie(s) worden afgeleid. Een eerste-orde systeem bereikt ≈99 % van de stapsprong in ongeveer 5 tijdconstanten (t_s ≈5τ)【33†L358-L364】. Rivera (1998) hanteert vergelijkbaar: bij PRBS-experimenten wordt aanbevolen de duur D ≥5τ_dominant te kiezen【101†L590-L598】. Concreet betekent dit dat de experimentele datasegmentlengte ten minste 5× de langste dominante τ moet zijn. (In resonante of ondergedempte systemen geldt ongeveer 4–5 keer de dempingstijd van het dominante paar polen.) Voor zekerheid kan men 5–6 tijdconstanten gebruiken. De Settling Time is dus leidend voor de segmentlengte: meet de stapsignaal tijd waarin het signaal stabiel wordt (binnen enkele % bandbreedte) en neem enkele malen die tijd als experimentele duur【33†L358-L364】【101†L590-L598】.  

Voor MIMO-systemen moet men in elk geexciteerd kanaal de laagst relevante frequentie (grootste τ) grijpen. Bij gelijktijdig aangestuurde X1,X2 (common mode) wordt de zwaarste massa (payload+X-motoren) verplaatst, dus de traagste modus in translation. In differentieel (yaw) mode is de traagheid lager, dus kortere τ. Voor veilig experiment: we kunnen in één datasegment eerst de common-modus-step afwachten (lengte ≈5τ_common) en daarna—eventueel na korte rust—een differential stap bijwerken (zie Workflow verderop).  

Een handige vuistregel is minstens 3–5 *periodes* van de laagste relevante frequentie te meten. Als de kleinste resonantiefrequentie f_min bijvoorbeeld 0,5 Hz is (τ≈0,3 s), zorg dan dat je minstens ~3–5 s data hebt. Rivera’s formules gebruikten aantal samples via D≥5τ_Hdom. In de praktijk betekent dit dat we na het commando wachten tot de snelheid (of fout) effectief nul is of tot de uitwijking minder wordt dan een fractie van de stapamplitude. Een logaritmische plot van de fout kan helpen om te zien wanneer de respons uitgeëbd is.  

【91†embed_image】 *Figuur: Typische ondergedempte staprespons van een tweedegraads systeem (ζ≈0.5), met piekovershoot M_p en opkomsttijd t_r【119†L139-L148】.*  

Uit de stapcurve bepaalt men demping (overshoot) en tijdconstanten. Een standaardondergedempte respons (figuur hierboven) laat piekovershoot M_p en een exponentiële afname tegen exp(–ζω_n t). Door M_p en de tijd tot de eerste top t_p kan men dempingsfactor ζ en hoekfrequentie ω_n schatten. Voor ons gantry gelden doorgaans lage ζ (ondergedempt). Bij ondergedempt gedrag schat men bijvoorbeeld ζ uit de verhouding tussen t_p en de eentonige tijdconstante. Het is belangrijk dat gestopt en teruggezet gedrag onderscheiden wordt: als de respons asymptotisch nadert, is het een stabiele polenmodi. Een blijvende helling of drift wijst op een integrator of verstoring. Bijvoorbeeld bij Weiss et al. steeg de fout aanvankelijk, maar door de integrator eindigde die respons na ≈400 ms exact bij het target【124†L1309-L1313】. 

# Duur van het experiment (window length)  

Uit de traagste modus volgt de benodigde duur: zoals gezegd ≥5×τ. Een empirische benadering is het *settling time* t_s (bv. 2 % of 5 % criterium) te meten uit de staprespons, en daarvan minstens D≈5 t_s of (conservatiever) 6–8 t_s aanhouden. Rivera geeft D≥5τ_Hdom【101†L590-L598】; Seborg nota $(t_{settle}\approx5τ)$【33†L358-L364】. We kunnen τ_Hdom schatten uit een snelle eerste stapmeting of uit eerdere berekeningen. 

Als vuistregel meet je na de stap tot bijvoorbeeld 98 % van de verandering bereikt is. Als dat bijvoorbeeld 1 s duurt, neem dan 5–8 s data. Herhaal voor elk modussegment. Een alternatieve benadering is te kiezen op laagste frequentie f_min: meet minimaal 3–5 cycli, oftewel D≈5/f_min. 

# Aantal tijdsconstanten en perioden  

Voor betrouwbare schatting gebruikt men typisch 4–6 tijdsconstanten per modus. Bij een tweede-orde met dominante τ laat men ±5τ meekijken zodat de amplitude tot <1 % geslonken is. Voor periodiciteit geldt: uit 3–5 cycli kan men frequentiecomponenten goed vaststellen, meer is beter om aliasing/ruis te dempen. 

Bij MIMO waarbij combinaties dominerende modes bepalen, gelden de langste modes. García-Herreros gaf als voorbeeld dat X- en Y-assen tot ±60 mm mogen uitkoppelen (yaw ±0,1 rad)【91†L663-L671】 – dit stelt een ondergrens aan de minimale signaalduur indien zo’n bewegingstijd significant is. 

# Integrator, wrijving en drift  

Men moet onderscheid maken tussen werkelijke plantdynamiek en effecten van de regelaar of niet-lineair gedrag. Een blijvende offset of langzaam naderen naar het target kan twee oorzaken hebben: de integrator in de controller (elimineert stf.-fout) of een traagvoudig proces. In Weiss et al. bleven de PI-regelaarresponsen op den duur exact op nulvout, terwijl de onbestuurde plant een overshoot en dan uitzweven zou vertonen【124†L1305-L1313】【124†L1309-L1313】. Dus als na de stap het systeem na verloop van tijd *precies* op het streefpunt uitkomt, is vaak de integrator de schuld. 

Om integrator-effect te isoleren, kan men de controller tijdelijk inactiveren (rust-zet-opnieuw tests) of in open loop meten (indien mogelijk). Wrijving uit zich in asymmetrie bij positieve/negatieve verplaatsingen: zorg voor stappen heen en terug om coulomb vs veldwrijving te zien. Frictie kan ook worden geïdentificeerd met constante-snelheid tests (zoals García-Herreros deden). 

Drijvende transducers (temp.-drift) en ruis veroorzaken langzaamere trends of instabiliteit in de gemeten output. Om die te onderkennen kan men nulbeweging uitvoeren: activeer de regelaar, maar geef geen doelverandering en bekijk of het signaal langzaam wegtijlt (drift) of ruisachtig beeft. Bij twijfel filter tijdelijke drift uit (smoothing) of herhaal tests om consistente dynamica te vinden. 

# Veilige excitatie binnen limieten  

Beschermactuatoren en stage: beperk stapomvang, snelheids- en accel.-capabilities. Weiss e.a. melden max X-snelheid 2 m/s en accel. 20 m/s²【91†L663-L671】. Ontwerp stapsignalen daarom als gemaximaliseerde rampen: houd start/vertragen progressief (S-vormige profielen) om jerk te beperken. Controleer mechanical stop-limieten: X1–X2 mogen niet meer dan ±60 mm relative bewegen (yaw ±0,1 rad)【91†L663-L671】. Gebruik softwareblokken (“slew rate limit”) of hardware-rem om overmatige versnelling te voorkomen. Let op temperatuurstijging bij herhaalde stappen.  

Clutching (ontkoppelen) kan gebeuren via servo-commando (motorremmen) of mechanisch klemsysteem. Bij het “clutchen” van een X-as kan men de motor uitkoppelen en de andere as stappen laten rijden, wat puur yaw exciteert. Doe dit steeds bij ongeveer nul-verplaatsing om botsing/slijtage te voorkomen.  

# MIMO-specifieke overwegingen  

De gantry heeft dubbele X-aandrijving: common mode (gelijke input) stuurt X-translatie, differential mode stuurt yaw (rotatie). Experimenteer daarom apart in deze modi: 1) **Common X-tests:** gelijke steps op X1 en X2 (of voer X1+X2 verst in commando) om translatie dynamiek te meten. 2) **Yaw-tests:** zet X1 en X2 in tegengestelde richting (clutchen één as en stap de andere) om yaw-Modus te prikkelen. 3) **Y-tests:** terwijl X1=X2 geblokkeerd zijn, stap Y-as. 

Zo ontkoppel je de bewegingsmodi. García-Herreros et al. tonen dat door X1=X2 synchroon te bewegen, je de sommassa (m1+m2+…+payload) vindt【91†L687-L694】. De differentiële vrijheid (yaw) geeft een veel hogere natural freq (lagere traagheid), zodat deze sneller settles; je hoeft de data niet zo lang te meten als voor common mode.  

Omdat Y-positie de massa-inertia matrix van X verandert, is dit LPV-achtig: repeteer de stappen op een paar verschillende Y-waarden (bijv. laag, midden, hoog) om de invloed van Y op X- en yaw-dynamiek te vangen. Gebruik zogeheten *bevroren* (frozen) tests: houd Y vast op een constante waarde tijdens een stap-experiment. Zo leer je lokale LTI-dynamiek. Er is geen strikte literatuurverwijzing nodig voor deze praktische aanpak, maar dit komt overeen met de “lokale LPV” aanpak in de literatuur: voer experimenten uit bij meerdere instellingen van de schedulingvariabele om het bereik te bestrijken. 

# Traject- vs. stappenexcitaties  

Stap- of clutchexperimenten brengen uitsluitend laagfrequente en lineaire modes in beeld; ze zijn simpel en beheersbaar. Echter, dynamische resonanties of hoge frequenties kunnen onopgemerkt blijven. Langzaam opvolgende stappen of trapezoïdale bewegingen (profilen) genereren bredere spectrumexcitaties en imiteren realistische bewegingen. 

García-Herreros toont dat stapsignalen heel wat informatie opleveren (massabalans, motorfrictie), maar dat uiteindelijk een trajecttest (bijv. een cirkel) de performance beter characteriseert【124†L1314-L1322】. Weefselsgewijs: als de voorgestelde operatie al bewegingen kent (bv. lineraire ramprijtrajecten), bekijk dan of die de traagste responspieken oproepen. Zo ja, kunnen we leren: het logged railvolgen bevat al honderden kleine versnellingen. Als niet, voeg dan gerichte stapsignalen toe. 

Als checklist: 
- Zijn alle relevante modi geprikkeld? (X, yaw, Y, bij verschillende Y-waarden)  
- Zijn input-signalen breedbandig genoeg? (één herhaalde stap exciteren vooral lage frequentie)  
- Neemt de gemeten output alle dynamiek over het belangwekkende bandbreedte op?  
- Staan de steady-state en transiënten vast (filters en drift beheerst)?  
- Past de verkregen data bij een eenvoudig lage-orde model?  

Indien trajecten weinig excitatie lijken te geven (bv. geen draaien of snelle versnellingen), is extra multisine of PRBS nodig. Zo’n multisine moet de lage frequenties beter bestrijken. Maar begin met stap-/ramptests voor grove parameterinschatting en modelvalidatie. Als trajectdata na deze controles nog steeds onvoldoende de langzame/polonale dynamiek onthullen, ga dan naar meer complexe excitatie.  

# Praktische workflow en regels  

**Stapsgewijze procedure:** 
1. *Voorbereiding:* Meet of bereken vooraf de actuatorlimieten (max. snelheid, accel, yawhoek). Definieer veilige amplitudes.  
2. *As-isolatie:* Ontwerp experimenten per DOF: (a) Blokkeer X1=X2 om Y-assen te testen (lagere X-koppeling, test Y). (b) X1=X2 bewegen (common X). (c) X1=-X2 bewegen (clutched diff. voor yaw). Gebruik lage amplitudes om lineair gebied te garanderen.  
3. *Geleidelijke stappen:* Voer kleine stappen (bijv. enkele millimeters) om eerste tijdconstante(r) te schatten. Controleer respons (overshoot, tijdconstante). Vergroot stapsignaal indien nodig, let op niet-lineaire effecten.  
4. *Snellere ramps:* Herhaal met stapsprongen tot limieten, observeer integratorgedrag. Let op residueel (na vastlopen) om frictie te meten.  
5. *Herhaal op meerdere posities:* Zet Y-index vast op verschillende posities (min–max) en herhaal X-tests. Dit levert LPV-variatie.  
6. *Data-analyse:* Bepaal dominante τ uit afklingtijden. Evalueer settling time; pas D≥5τ toe. Controleer modelresiduen (bijv. fits van simpele FOPDT/SOPDT).  

**Segmentlengte:** Neem van elke staprespons voldoende lange meting, ruwweg 5–6 τ. In de tijdserie: blijf meten totdat de curve qua amplitude bijna is uitgezwerkt (bijv. <2 % van de stapsprong). Gebruik (semi-)logaritmische schaal van de fout om uitschieters op langere tijden op te merken.  

**Aanbevolen plots:** 
- Time-series van positie (of fout) voor elke stap: marker overshoot, time-to-settle, eindsignaal (zie figuren hierboven).  
- Zoom plots in log-schaal van de helft hoogte (toont tijdconstante).  
- Bode/Spectrum: indien mogelijk alg. freq-resp uit de stap door integrale transformatie (voor validatie laagste freqs).  
- MIMO: vergelijk respons van X1 vs X2 bij gelijke commando’s (symmetrie).  
- Evaluatie: afbeeldingsplots van fitted model vs gemeten stap.  

**Checklijst Traject-** of **multisinusbehoefte:**  
- **Dekken gestapelde modi?** Als een (traject)commando alleen gerichte bewegingen (bv. enkel X) bevat, prikkelt het yaw niet. Voeg differential test toe.  
- **Frequentiespec:** Als het traject hoofdzakelijk laagfreq is, kunnen hogere (overblijvende) dynamica missen. Meet met sinusdumpes of prbs op die bandbreedte.  
- **Lineair gedrag:** Als stapresponspunten sterk amplitude-afhankelijk zijn (veel overshoot bij klein, weinig bij groot zoals in [124]), dan moet men opletten. Grote amplitudes prikkelen wrijving meer, dus gebruik meerdere niveaus voor ID.  
- **Controle residuen:** Voorspelt een model getraind op trajectdata het stapgedrag goed? Zo niet, dan mis je excitatie. Als ja, dan is wellicht multisinus niet nodig.  

# Belangrijkste bronnen (aanraders)  

1. García-Herreros *et al.*, *“Model-based decoupling control method for dual-drive gantry stages: A case study”* (Control Eng. Pract., 2013) – Bespreekt expliciet identificatie van een gantry met gekloofde X-as. Legt uit hoe met slotten en constante-bewegingstesten de massa’s, fricties en yaw-stijfheid worden bepaald【91†L677-L686】【91†L687-L694】. Essentieel voor onze experimentwijzen.  
2. Weiss *et al.*, *“Closed-Loop Control of a 3D Printer Gantry”* (ACC 2020) – Toont praktijkvoorbeelden van staptesten in gesatureerde servo-omgeving. Illustreert hoe stappen leiden tot overshoot en hoe de PI-regelaar de system performance beïnvloedt【70†L1272-L1275】【124†L1309-L1313】. Goede illustraties van stapgedrag en limieten.  
3. Seborg *et al.*, *“Process Dynamics and Control”* – Lecture notes (of een systemidentificatietekst): behandelt basale stap- en overgangsrespons en geeft vuistregels zoals t_s≈5τ voor settling【33†L358-L364】. Handig voor ontwerpregels.  
4. Rivera (ASU SysID course, 1998) – Slides over gesloten-lus system ID en experimentontwerp. Geven formules voor experimentduur (D≥5τ)【101†L590-L598】 en behandelen multipele ingangen (PRBS).  
5. Liu *et al.*, *“A tutorial review on step or relay feedback test”* (J. Process Control, 2013) – Survey over stap- en relais-ID in industrie, inclusief gesloten-lus adviezen. Handig voor achtergrond (bibliografie, methoden), al citeerde Weiss hieruit ideeën over integratoren.  

Deze bronnen geven gezamenlijk praktische richtlijnen voor het ontwerpen van stap-/clutchexperimenten en het schatten van traagste dynamiek in een gesloten-lus gantry-systeem. Gebruik de stapresponsdata om dominant poles en tijdconstanten te schatten, experimenteer lang genoeg (≈5–6 τ), en verifieer met simulatie of fits. Vergeet niet de controller (integraalterm) en niet-lineaire effecten (wrijving, verzadiging) te herkennen in de responsen. De hierboven beschreven workflow, segmentbepaling en checklist bieden een leidraad voor een systematische experimentele aanpak.  

