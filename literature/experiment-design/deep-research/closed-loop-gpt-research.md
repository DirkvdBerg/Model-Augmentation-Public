# Gesloten-lus identificatie en fysische parameterterugwinning voor een dual-drive gantry

## De relevante signaalpaden in jouw opstelling

Voor jouw setup is de sleutelobservatie dat de extra multisine \(f_{\text{sim}}(t)\) **na** de regelaar wordt geïnjecteerd, dus als een plant-ingangsverstoring wordt gezien. Voor een gefixeerde scheduling-trajectorie \(Y(t)\) — dus lokaal gezien als LTI of “frozen/quasi-LPV” — krijg je voor één kanaal schematisch

\[
u(t)=C\bigl(r(t)-y(t)\bigr)+f_{\text{sim}}(t), \qquad y(t)=G\,u(t).
\]

Daaruit volgt direct

\[
u(t)=\frac{C}{1+GC}\,r(t)+\frac{1}{1+GC}\,f_{\text{sim}}(t)
      = CS\,r(t)+S\,f_{\text{sim}}(t),
\]

\[
y(t)=\frac{GC}{1+GC}\,r(t)+\frac{G}{1+GC}\,f_{\text{sim}}(t)
      = T\,r(t)+GS\,f_{\text{sim}}(t),
\]

met

\[
S=\frac{1}{1+GC}, \qquad T=\frac{GC}{1+GC}.
\]

Dit is precies de klassieke gevoeligheidsstructuur: plant-ingangsverstoringen worden met \(S\) naar de netto plantinput en met \(GS\) naar de output gevoerd, terwijl referenties via \(T\) naar de output gaan. In de “Gang of Four”-notatie wordt de respons op een load disturbance gegeven door \(G_{yd}=P/(1+PC)=PS\), en de referentierespons door \(G_{yr}=PC/(1+PC)=T\). Åström vat dat kernachtig samen: “Typically \(S(0)\) small and \(S(\infty)=1\) and consequently \(T(0)=1\) and \(T(\infty)\) small” (p. 27), en elders: “The effect of feedback is thus like sending the open loop output through a system with the transfer function \(S=1/(1+PC)\)” (p. 29). citeturn10view0

Daarmee is jouw diagnose theoretisch juist. **Onder de gesloten-lus bandbreedte**, waar \(|GC|\gg 1\), geldt \(|S|\ll 1\) en \(|T|\approx 1\). Dus een na-de-regelaar ingespoten krachtmultisine wordt dan grotendeels weggewerkt door de feedback, terwijl een referentietraject in diezelfde frequentieband juist goed naar de output wordt overgedragen. Forssell en Ljung formuleren het algemene gevolg zo: closed-loop data heeft typisch “less information about the open-loop system” omdat een doel van feedback juist is de lus **ongevoelig** te maken voor veranderingen in de open-lusdynamica. citeturn40view1turn10view0

Toegepast op jouw Y-as model

\[
G_Y(s)=\frac{1}{m_h s^2 + c_y s}=\frac{1}{s(m_h s+c_y)},
\]

ligt de informatie over \(c_y\) vooral in het lage-frequentie, snelheid-gedomineerde regime, terwijl hoge frequenties vooral \(m_h\) zichtbaar maken omdat dan de inertieterm domineert. Juist die lage frequenties zijn in jouw huidige injectieschema het meest onderdrukt door de kleine \(S\). Daarom is jouw werkhypothese — **lage-frequentie Y-krachtmultisine verwijderen en \(c_y\) vooral uit referentietrajectcontrast halen** — niet alleen praktisch maar ook theoretisch goed onderbouwd. citeturn10view0turn40view1

## Wat gesloten-lus identificatietheorie hierover zegt

De standaard closed-loop literatuur onderscheidt drie families: de **direct method**, de **indirect method** en **joint input-output / projection / two-stage** benaderingen. In de samenvattende colleges van entity["organization","Eindhoven University of Technology","eindhoven, nb, nl"] wordt de direct method samengevat als: gebruik de gemeten \(y\) en \(u\), identificeer een standaard prediction-error model, en negeer de aanwezigheid van \(C\) in de regressie. Onder de gebruikelijke voorwaarden — het ware systeem ligt in de modelset, geen algebraïsche lus, en de data zijn informatief — zijn de schattingen consistent en behouden ze de bekende maximum-likelihood/Cramér–Rao eigenschappen. Tegelijk zegt dezelfde samenvatting ook: “No ‘free’ excitation of input \(u\); periodic excitation of \(u\) is not feasible” (slide 24). citeturn25view0turn37view0

Dat laatste punt is belangrijk voor jouw vraag. In closed loop is de gerealiseerde input \(u\) **geen vrij ontworpen identificatiesignaal**: hij is het resultaat van de interactie tussen referentie, plant, controller en verstoringen. Daarom is een periodieke multisine die op de plant-som wordt gezet niet automatisch een periodieke netto plantinput; die netto input wordt met \(S\) vervormd. Dat is precies waarom jouw \(u_{\text{plant},Y}=u_{\text{fb},Y}+f_{\text{sim},Y}\) in de lage frequenties klein blijft ondanks een groot \(f_{\text{sim},Y}\). citeturn25view0turn10view0

Voor jouw **synthetische, ruisloze Simulink-dataset** is er echter ook goed nieuws. Forssell schrijft over de eenvoudige lineaire closed-loop situatie dat de ruisvrije regelwet op korte data exact kan worden bepaald en dat “\(r\) carries no further information about the system, if \(u\) is measured” (pp. 5–6). Met andere woorden: als jij in simulatie de **daadwerkelijke plantinput** \(u_{\text{plant}}\) exact reconstrueert en die gebruikt in je tijdsdomein-fit, dan zit je zeer dicht bij het gunstige regime waarin de direct method conceptueel schoon is. citeturn39view3

Die gunstige conclusie geldt **niet automatisch** op hardware. Van den Hof, Dankers en Weerts laten expliciet zien dat “the direct method loses consistency when correlated disturbances are present … or when sensor noises are present” (abstract/p. 1), en verder dat sensorruis op de node-signalen de klassieke direct method tot een errors-in-variables probleem maakt. Dat onderscheid tussen **synthetisch/noiseloos** en **hardware/ruisend** is in jouw thesis essentieel. citeturn21view0

De samenvatting voor jouw geval is daarom scherp af te bakenen. Voor jouw huidige parameter-recovery op synthetische \(q_1\)-data met exact gereconstrueerde \(u_{\text{plant}}\) is directe tijdsdomein-fitting van fysische parameters goed verdedigbaar. Voor hardware of Simscape-data met sensoren, inner loops, saturaties en actuatoronzekerheid moet je strenger zijn en eventueel naar instrumentele variabelen, projection/two-stage of joint input-output hulpmiddelen grijpen. citeturn21view0turn25view0

## Referentie-excitatie versus plant-inputverstoringen

Voor klassieke LTI-FRF-identificatie is de hoofdles uit de closed-loop literatuur dat je de plant **niet** naïef uit \(S_{uy}/S_{uu}\) mag halen zodra er feedback actief is. De open-loop college slides van entity["organization","Delft University of Technology","delft, zh, nl"] zeggen letterlijk: in gesloten lus geldt \(S_{un}(f)\neq 0\), en dus \(H(f)\neq S_{uy}(f)/S_{uu}(f)\). De remedie is een **externe** ingang \(r\) gebruiken, met \(S_{rn}=0\), en op basis daarvan een referentie-gebaseerde schatter op te bouwen. citeturn19view0turn30view3turn30view4

Dat is precies de gedachte achter de **indirect method**: schat de gesloten-lustransfers \(G_{yr}\) en \(G_{ur}\) uit de externe referentie naar respectievelijk output en input, en neem dan hun quotiënt. In de colleges van Van den Hof staat dit heel compact: “The transfers \(r\rightarrow [y\;u]^T\) can be estimated with open-loop methods” (slide 26), en als \(G_{yr}\) en \(G_{ur}\) consistent zijn, dan is \(\hat G = \hat G_{yr}/\hat G_{ur}\) een consistente plantschatting. Belangrijk voor jouw vraag is ook de designkant: “Any desired excitation signal can be used for \(r\) (e.g. periodic)” (slide 31). citeturn25view0turn31view9turn32view1

De **projection / two-stage** gedachte gaat nog directer naar jouw probleem. Daar wordt de input opgesplitst als

\[
u(t)=u^r(t)+u^e(t),
\]

waar \(u^r\) het door de externe referentie veroorzaakte deel is en \(u^e\) het door ruis/verstoringen veroorzaakte deel. Vervolgens identificeer je \(G_0\) op basis van \(u^r\) en \(y\); de bron zegt daar letterlijk over: “This is basically an open-loop problem” (slide 30). citeturn32view0

Voor mechatronische systemen is deze voorkeur voor referentie-excitatie ook praktisch terug te vinden. In een paper over closed-loop FRF-bepaling bij industriële manipulators schrijven Saupe en Knoblach: “It is usually preferred to insert the excitation as a reference signal … instead of a disturbance in the input of the plant” (p. 2). Hun reden is niet dat plant-ingangsverstoringen principieel verboden zijn, maar dat de gesloten-lus trackingconfiguratie de werkelijk gerealiseerde plantinput bepaalt en dat je die dus beter doelbewust via \(r\) aanstuurt dan indirect via een disturbance-kanaal dat de controller zal proberen te onderdrukken. citeturn22view0

Voor consistentie en informativiteit is bovendien niet alleen “persistent excitation” van belang, maar het bredere begrip **data informativity**. De Lyon-closed-loop colleges formuleren dat als een voorwaarde op de spectraalmatrix

\[
\Phi_z(\omega)=
\begin{bmatrix}
\Phi_u(\omega) & \Phi_{uy}(\omega)\\
\Phi_{yu}(\omega) & \Phi_y(\omega)
\end{bmatrix},
\]

met als sterk LTI-geval \(\Phi_z(\omega)>0\) voor alle \(\omega\). Die colleges zeggen expliciet: “Rather than \(r\) being persistently exciting, it is sufficient to require that the data set is informative with respect to \(\mathcal M\)” (slide 17). Dit is precies waarom **lijnentelling alleen** geen bewijs van parameter-identificeerbaarheid is; informativiteit is een eigenschap van de data **relatief aan de modelstructuur**, niet alleen van het excitatie-signaal. De LPV-literatuur generaliseert datzelfde punt expliciet van LTI naar LPV/LPV-ARX modelstructuren. citeturn25view0turn31view8turn32view2turn26search0

Voor MIMO geldt een vergelijkbare nuance. De multivariabele FRF-schatter \(\hat H=\hat S_{zy}\hat S_{zu}^{-1}\) bestaat alleen als \(\hat S_{zu}\) volle rang heeft op de relevante frequenties. Dat ondersteunt jouw keuze om te denken in **common**, **diff** en **y**-modi: dat is fysisch zinvol, maar de verzameling experimenten moet samen wel de relevante subruimte op volle rang exciteren. Tegelijk laat de MIMO closed-loop literatuur zien dat het **niet noodzakelijk** is alle referentiekanalen tegelijk te exciteren voor identificeerbaarheid, maar dat excitatie van meer referenties de nauwkeurigheid “never worsens and, in most cases, improves” van de schattingen. citeturn19view0turn30view5turn30view6turn6search9turn20search0

De bottom line voor jouw vraag is dus: **voor lage frequenties onder de gesloten-lus bandbreedte is referentie-excitatie in jouw setup veel informatiever dan een na-de-regelaar geïnjecteerde krachtmultisine.** citeturn10view0turn22view0turn32view0

## Wat de literatuur zegt over multisines in gesloten lus

Jouw huidige multisine-ontwerp sluit op veel punten al goed aan bij de literatuur. Periodieke excitatie op integer-harmonische lijnen is klassiek verdedigbaar, omdat bij periodieke excitatie de FRF “by simple division of the output by the input spectrum” kan worden bepaald, en omdat periodische, deterministische multisines leakage vermijden. In een toepassingspaper over multisine-FRF-identificatie staat zelfs letterlijk dat dergelijke signalen “offer substantial advantages (e.g. avoiding leakage in the frequency domain data)” (p. 2). citeturn14view0turn14view3

Ook je keuze voor **Schroeder-fases** is goed te verdedigen. Vyncke et al. schrijven dat Schroeder-fases “good results” geven voor vlakke, bandbegrensde multisines met opeenvolgende geëxciteerde frequenties, en rapporteren dan typisch een crest factor van ongeveer 1.7; random fases zijn slechter, en numerieke optimalisatie kan nog iets beter maar is zwaarder. Dat past precies bij jouw keuze voor periodieke 1-seconde periodes, integer lijnen, per-kanaal RMS-normalisatie en praktisch uitvoerbare crest-factorbeheersing. citeturn14view3

Het gebruik van **odd harmonics only** verdient een subtielere duiding. Voor een lineair, synthetisch, ruisloos parameter-recovery probleem is odd-only niet fundamenteel nodig. De literatuur gebruikt odd random-phase multisines vooral om **nietlineariteiten** te detecteren en te scheiden van de lineaire bijdrage. Een VUB-proefschrift formuleert dat expliciet: “The solution to detect both even and odd nonlinear contributions is to use an excitation set only with odd harmonics” (p. 6). Een latere closed-loop FRF-paper van Pintelon en Schoukens laat zien hoe odd random-phase multisines met random harmonic grids de BLA, ruisvariantie en even/odd distortion levels kunnen blootleggen, ook in feedback, mits de input-signal-to-distortion-ratio groot genoeg blijft. citeturn17view1turn17view0

Daarmee volgt voor jouw thesis een belangrijke scheiding. In het **hoofdstuk over parameter recovery** kun je odd-only beschrijven als een praktische ontwerpkeuze die compatibel is met goede crest factor en nette periodieke spectra. In een **diagnostisch hoofdstuk** voor latere hardware-relevantie kun je much sterker zeggen dat odd-only en eventueel random skipped lines nuttig zijn om nonlinear distortions, harmonische vervuiling en BLA-validiteit te beoordelen. Die dubbele framing is inhoudelijk correcter dan doen alsof odd-only intrinsiek vereist is voor het huidige ruisloze simplified-model probleem. citeturn14view3turn17view1turn17view0

Voor MIMO-multisines ondersteunt de literatuur jouw modalebasis-benadering, mits de gerealiseerde inputmatrix op de geëxciteerde frequenties voldoende onafhankelijk blijft. Het Delft-materiaal stelt expliciet dat \(\hat H\) alleen bestaat als \(S_{zu}\) invertibel of full-rank is. Dat betekent voor jouw common/diff/y basis niet dat elke run alle drie kanalen simultaan moet exciteren, maar wél dat de **verzameling** experimenten de relevante richtingen moet dekken, en dat je bij de evaluatie naar de **gerealiseerde** \(u_{\text{plant}}\) moet kijken, niet alleen naar het ontworpen \(f_{\text{sim}}\). citeturn19view0turn30view6

Voor closed-loop FRF-validatie is bovendien **coherentie** nog steeds nuttig, maar dan op de juiste signalen. In closed loop is het zinloos om een naïeve open-loop-coherentie op \(u\) en \(y\) te interpreteren alsof de lus niet bestaat. De gesloten-lus FRF-notes adviseren daarom coherenties te bekijken die horen bij de gevoeligheids- of referentie-gebaseerde schatters; ze noemen coherentie expliciet een nuttige quality indicator, maar waarschuwen ook dat je moet weten **wélke** FRF je aan het schatten bent. citeturn28view0turn30view4

## Consequenties voor jouw quasi-LPV mechanische parameterfit

Hier moet je in je thesis heel zorgvuldig drie niveaus uit elkaar houden.

Ten eerste is er de **klassieke LTI closed-loop FRF-identificatie**. Die vertelt je hoe een extern referentiesignaal, een plant-ingangsverstoring en de controller samen de spectra van \(u\) en \(y\) vormen, en waarom referentie-gebaseerde of sensitivity-gebaseerde FRF-schatters nodig zijn in gesloten lus. Dat niveau is precies het juiste niveau om te verklaren waarom jouw lage-frequentie Y-force multisine onderdrukt wordt. citeturn10view0turn19view0turn25view0

Ten tweede is er jouw **werkelijke identificatieprobleem**: tijdsdomein-parameterterugwinning van fysische parameters in een vereenvoudigd quasi-LPV mechanisch model,

\[
M(Y)\ddot q + C\dot q + K q = u,
\]

waarbij \(Y\) tegelijk output/toestand én schedulingvariable is. De LPV-literatuur benadrukt dat informativiteit en identificeerbaarheid daar algemene begrippen blijven, maar niet simpel terugvallen op een gewone transferfunctie-redenering; de scheduling-trajectorie maakt deel uit van het informatiegehalte van de data. Daarom zou het onjuist zijn om te zeggen dat “genoeg multisine-lijnen” op zichzelf de fysische parameteridentificeerbaarheid bewijzen. In jouw probleem hangen \(m_h\), \(c_y\) en eventuele koppelingen af van de **gerealiseerde** bewegingsregimes en scheduling-trajecten, niet alleen van het nominale lijnenspectrum. citeturn26search0turn25view0turn31view8turn32view2

Ten derde is er het onderscheid **synthetisch/noiseloos versus hardware/ruisend**. In jouw huidige dataset zijn scheduling \(Y(t)\), output \(q_1(t)\) en de gereconstrueerde input \(u_{\text{plant}}(t)=u_{\text{fb}}(t)+f_{\text{sim}}(t)\) exact bekend uit simulatie. In dat regime is het volkomen verdedigbaar — en eigenlijk noodzakelijk — om de identificatiemodelinput gelijk te nemen aan de **daadwerkelijk gerealiseerde plantinput**. Als \(f_{\text{sim}}\) werkelijk in de data-generatie aanwezig was, dan zou het weglaten ervan in het identificatiemodel een fysisch verkeerde inputdefinitie opleveren en de parameterfit dwingen die fout via \(M,C,K\) te compenseren. De direct-method literatuur ondersteunt precies dit gebruik van gemeten \(u\) en \(y\) in het gunstige, ruisvrije closed-loop geval. citeturn39view3turn25view0turn31view7

Daarmee kom je ook bij je hoofdvraag over \(c_y\) en \(m_h\). Voor jouw Y-as is een lage-frequentie na-de-regelaar krachtmultisine bijna per definitie een **slechte** bron van extra informatie over \(c_y\), omdat die via \(S\) de lus in moet en dus juist in het lage-frequentiegebied wordt weggecorrigeerd. De juiste plaats om \(c_y\) zichtbaar te maken is dan niet een disturbance-achtige injectie onder de bandbreedte, maar een **referentiegedreven trajectcontrast** dat verschillende verhoudingen van snelheid en versnelling afdwingt. Precies dat doen jouw T1 en T6 in concept: T1 levert relatief meer snelheid-/dempingsgevoeligheid, T6 relatief meer versnelling-/inertiegevoeligheid. Dat is theoretisch veel beter uit te leggen dan blijven vasthouden aan een lage-frequentie Y-force multisine die de gesloten lus zelf neutraliseert. citeturn10view0turn22view0turn32view0

Een hoge-frequentie Y-multisine kan nog steeds zin hebben, maar dan als **secundaire** perturbatie voor inertiële informatie, niet als primaire bron voor \(c_y\). Boven de bandbreedte nadert \(S\) naar 1, zodat de injectie meer “overleeft”, maar jouw eigen model impliceert dan ook dat de inertieterm domineert en demping minder goed scheidbaar wordt. Bovendien waarschuwt de mechatronica-literatuur in dat gebied voor hogere gevoeligheid voor unmodeled resonances, drive/load limits en slechtere bruikbare SNR. Kortom: boven-bandbreedte injectie is een logische oplossing voor “survival through the loop”, maar geen wondermiddel voor lage-frequentie dempingsidentificatie. citeturn10view0turn22view0

Voor de X-common en X-diff multisines geldt dezelfde logica, maar niet per se dezelfde uitkomst. **Hou ze alleen als de gerealiseerde netto plantinput en outputrespons aantoonbaar niet door de gesloten lus worden weggefilterd in de gekozen band.** Als die multisines in jouw diagnostieken wél als echte additionele excitatie zichtbaar blijven in \(u_{\text{plant}}\), dan zijn ze nuttig voor common/differential/rotationele koppelingen. Als ze in hun lage-frequentiecomponenten net zo hard door \(S\) worden onderdrukt als de Y-injectie, dan moet je ze precies zo behandelen: omhoog schuiven in frequentie, amplitude terugbrengen, of helemaal schrappen. citeturn10view0turn30view6

## Aanbevolen experimenteel ontwerp en scriptietekst

Mijn praktische aanbeveling voor **jouw exacte setup** is als volgt.

**Verwijder de Y-as after-controller force multisine in 1–20 Hz als primaire identificatie-excitatie.** In jouw gesloten-lus architectuur is dat een plant-ingangsverstoring, en dus wordt de effectieve plantinput in dat gebied met \(S\) geschaald. Juist daar is \(S\) klein en zit jouw \(c_y\)-informatie. Theoretisch is dit dus de minst gunstige plek om extra Y-informatie te zoeken. citeturn10view0turn22view0

**Gebruik referentietrajecten als primaire excitatiebron voor parameter recovery**, vooral voor \(c_y\) versus \(m_h\). Dat past zowel bij de indirect/projection closed-loop theorie als bij jouw modelstructuur: via \(T\) wordt lage-frequentie referentie goed doorgegeven, en via trajectcontrast kun je snelheid- en versnellingseffecten scheiden zonder te vertrouwen op een disturbance-kanaal dat de controller zelf compenseert. citeturn10view0turn25view0turn32view0

**Hou after-controller multisine alleen als secundaire perturbatie**, en alleen in kanalen/banden waar de injectie de lus aantoonbaar overleeft. Practisch betekent dat: selecteer frequenties waar de gerealiseerde verhouding \(|U_{\text{plant},f}(j\omega)|/|F_{\text{sim}}(j\omega)|\) niet verwaarloosbaar is, dus waar \(|S(j\omega)|\) niet heel klein is. Voor de Y-as zal dat waarschijnlijk vooral relevant zijn voor hogere frequenties en dus eerder voor \(m_h\) dan voor \(c_y\). citeturn10view0turn37view0

**Behoud X-common en X-diff multisines conditioneel, niet automatisch.** De common/differential modale basis is fysisch heel logisch voor een dual-drive gantry, maar de literatuur eist uiteindelijk volle-rang/genoeg onafhankelijke gerealiseerde inputinformatie, niet alleen een nette ontwerpintentie. Laat dus de net gerealiseerde \(u_{\text{plant}}\), referentie-gebaseerde FRF’s en coherenties beslissen of ze echt informatief zijn. citeturn19view0turn30view6turn20search0

**Beperk amplitudes op basis van de nominale gesloten-lus bedrijfsconditie, niet op basis van hardware-maxima.** Voor de huidige synthetische fysische parameterfit is het doel niet “maximaal duwen”, maar een perturbatie ontwerpen die genoeg additionele informatie levert zonder de parameterrecovery te laten domineren door een kunstmatige disturbance-experimentconditie. De robotliteratuur adviseert expliciet conservatieve initiële excitatie en gebruikt de plantinput als leidende grootheid voor belasting/experimentdesign. citeturn22view0

**Voeg aparte, optionele “diagnostische” experimenten toe voor nette FRF/coherence-plots.** Denk aan korte referentie-multisine runs of ID-hold/frozen-\(Y\) segmenten, juist niet als hoofdbron voor de quasi-LPV parameterfit, maar voor verifieerbare LTI-diagnostiek: \(G_{yr}\), \(G_{ur}\), sensitivity/process sensitivity, en referentie-gebaseerde coherentie. Dat maakt je thesis methodologisch veel sterker, omdat je dan expliciet laat zien wat in de gesloten lus overblijft van je ontworpen excitatie. citeturn25view0turn30view4turn28view0

Een compacte, verdedigbare thesisformulering voor jouw huidige studie zou ik als volgt schrijven:

> In de parameter-recovery experimenten is de primaire excitatie aangebracht via het referentietraject, niet via een lage-frequentie krachtmultisine op de plantingang. De reden is dat een na-de-regelaar geïnjecteerde kracht in gesloten lus als een plant-ingangsverstoring werkt en daarom met de gevoeligheidsfunctie \(S=(I+GC)^{-1}\) naar de netto plantinput wordt gevormd. Onder de lusbandbreedte is \(|S|\) klein, zodat lage-frequentie krachtinjectie grotendeels door feedback wordt onderdrukt, terwijl referentiegedreven bewegingen in hetzelfde gebied via de complementaire gevoeligheid \(T=GC(I+GC)^{-1}\) juist wel effectief worden overgedragen. Daarom is voor de Y-as de scheiding tussen demping \(c_y\) en inertie \(m_h\) voornamelijk gerealiseerd met trajectcontrast tussen snelheid- en versnelling-gedomineerde referentiemanoeuvres, terwijl after-controller multisine hoogstens als secundaire perturbatie is gebruikt in frequentiebanden waar de injectie aantoonbaar de gesloten lus overleeft. Voor de identificatiemodellen is steeds de daadwerkelijk gerealiseerde plantinput \(u_{\text{plant}}=u_{\text{fb}}+f_{\text{sim}}\) gebruikt, zodat de inputdefinitie consistent blijft met de data-generatie. Deze argumentatie is gebaseerd op klassieke closed-loop identificatietheorie voor LTI-systemen, maar wordt in deze studie uitsluitend gebruikt als lokale/diagnostische onderbouwing; globale fysische parameteridentificeerbaarheid in het quasi-LPV model wordt niet afgeleid uit persistent-excitation lijnentelling alleen, maar uit de informativiteit van de gerealiseerde tijdsdomeintrajecten. citeturn10view0turn25view0turn32view0turn39view3turn26search0

Mijn eindadvies in één zin: **ja, verwijder de lage-frequentie Y after-controller multisine; gebruik referentietrajecten als hoofdexcitatie, hou after-controller multisines alleen waar ze de lus aantoonbaar overleven, gebruik altijd \(u_{\text{plant}}\) als modelinput, en voeg aparte referentie-gebaseerde FRF/coherence-experimenten toe voor een methodologisch schone thesis.** citeturn10view0turn22view0turn25view0turn28view0