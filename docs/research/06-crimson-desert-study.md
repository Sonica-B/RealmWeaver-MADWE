# 06 — Crimson Desert (Pearl Abyss) — Granular Game Study

**Scope.** Everything a planner needs to scope a world/combat/systems-heavy open-world action game
against a 2026 shipped reference: facts, world, player character, combat, systems, narrative,
presentation, scale metrics, design pillars, comparables, and a flat systems inventory.

**Method.** Web search (standard + extended) and page fetches performed on 2026-10-06. Sources:
Wikipedia, the Steam store page, Pearl Abyss official pages and notices, Inven Global's coverage
of Pearl Abyss's GDC 2025 / Gamescom Dev 2025 / CEDEC 2026 talks, major reviews (IGN, PC Gamer,
Eurogamer, GameSpot, Game Informer, Destructoid, PCGamesN, Push Square, Vice, Kotaku/TheGamer
round-ups), Gamescom 2024 hands-ons, patch notes, and community wikis/guides. Every bullet carries
a URL. IGN, Eurogamer, Rock Paper Shotgun, VG247 and several PC Gamer pages could not be fetched
directly (blocked); their positions are cited through round-ups that quote them. **No Rock Paper
Shotgun or Polygon review was located** (see §9).

**Confidence legend.** Unmarked = stated by an official source or by two or more independent
outlets. `[single-source]` = one guide/wiki only. `[uncertain]` = conflicting sources or
community-derived numbers that may drift with patches. `[pre-release]` = from previews/talks, may
not match shipped build.

---

## 1. Facts

### 1.1 Developer, publisher, release, platforms, price

* Developer and publisher: **Pearl Abyss** (Black Desert Online studio), self-published.
  https://en.wikipedia.org/wiki/Crimson_Desert
* Release date: **19 March 2026**, simultaneous on PS5 / PS5 Pro, Xbox Series X|S, Windows (Steam,
  Epic Games Store), macOS (Steam, Mac App Store) and GeForce NOW.
  https://en.wikipedia.org/wiki/Crimson_Desert ,
  https://crimsondesert.pearlabyss.com/ ,
  https://www.vgchartz.com/article/465822/crimson-desert-launches-march-19-2026-for-ps5-xbox-series-xs-and-pc/
* Nintendo Switch 2 version planned for **early 2027**. https://en.wikipedia.org/wiki/Crimson_Desert
* Release date was announced at Sony's **State of Play, September 2025**; the game "went gold" before
  launch after ~6–7 years of development.
  https://www.pushsquare.com/news/2025/09/stupidly-ambitious-ps5-open-world-crimson-desert-nails-down-march-2026-release-date ,
  https://www.gosugamers.net/entertainment/news/77883-crimson-desert-goes-gold-after-six-years-of-development-set-to-launch-in-march-2026
* Steam app id **3321460**; store page now titled "Crimson Desert Enhanced" after the free 2.0
  update. https://store.steampowered.com/app/3321460/Crimson_Desert/
* Price: Standard **$69.99**, Deluxe **$79.99** (Steam lists the Deluxe Pack DLC separately at
  $12.99), Collector's Edition **$279.99** (17-inch Kliff vs Golden Star diorama, SteelBook, fabric
  map, brooch, photo cards, patches; in-game Ultimate Pack + Deluxe Pack). Pre-order bonus: Khaled
  Shield; PS5-exclusive Grotevant Plate Set. One secondary source lists Deluxe at $89.99
  `[uncertain]`.
  https://store.steampowered.com/app/3321460/Crimson_Desert/ ,
  https://insider-gaming.com/crimson-desert-pre-order-bonuses-editions-prices/ ,
  https://www.pushsquare.com/news/2025/09/good-god-crimson-deserts-ps5-collectors-edition-costs-a-whopping-usd280
* Single-player only; Pearl Abyss states it is "not designed as a live-service experience" and has
  "a defined beginning and end". Multiplayer "not ruled out" but undecided.
  https://www.gamereactor.eu/we-talk-difficulty-scale-and-inspirations-in-crimson-desert-with-pearl-abyss-1674833/ ,
  https://mp1st.com/news/crimson-desert-multiplayer-not-ruled-out-by-pearl-abyss-still-undecided-on-mod-support
* Languages: 15 full (interface/audio/subtitles) incl. English, French, Italian, German, Spanish
  (ES/LatAm), Japanese, Korean, Polish, Russian, Chinese (S/T), PT-BR, Turkish. The 2.0 "Enhanced"
  update (26 Aug 2026) added **voice-over** in German, French, Spanish, PT-BR and Japanese, and
  Arabic UI/subtitles.
  https://store.steampowered.com/app/3321460/Crimson_Desert/ ,
  https://crimsondesert.pearlabyss.com/en-US/News/Notice/Detail?_boardNo=125 ,
  https://www.gematsu.com/2026/08/crimson-desert-enhanced-update-now-available

### 1.2 Development timeline and team

* Development started **early 2018**; ~**7 years** total; originally conceived as a Black Desert
  prequel/MMO, became a standalone single-player title during development.
  https://en.wikipedia.org/wiki/Crimson_Desert
* Code name "Project CD" (early 2019); announced as Crimson Desert at **G-Star 2019**; first
  gameplay reveal at The Game Awards 2020.
  https://en.wikipedia.org/wiki/Crimson_Desert ,
  https://mmoculture.com/2020/12/crimson-desert-pearl-abyss-interview-clarifies-single-player-and-multi-player-questions-and-more/
* Delay history: Winter 2021 target → indefinitely delayed July 2021 → Gamescom ONL 2023 re-reveal
  (no date) → analysts moved it from Q2 2024 to Q2 2025 → "2025" → delayed "one quarter" to Q1 2026
  (Aug 2025 earnings call) → 19 Mar 2026.
  https://www.dsogaming.com/news/crimson-desert-has-been-reportedly-delayed-until-2025/ ,
  https://wccftech.com/crimson-desert-reportedly-moved-to-q2-2025-dokev-to-2026/ ,
  https://www.pcgamer.com/games/action/crimson-desert-is-being-unavoidably-delayed-for-a-second-time-now-set-to-release-in-early-2026/ ,
  https://www.gematsu.com/2025/08/crimson-desert-delayed-to-q1-2026
* Team size: **~200 people**, "never exceeded 300" at peak (Design Office leads Doo Seung-bin and
  Kim Hyun-kyum, CEDEC 2026). Executive producer: founder **Daeil Kim**.
  https://www.invenglobal.com/articles/24095/pearl-abyss-unveils-crimson-desert-development-process ,
  https://otakukart.com/crimson-desert-devs-say-their-200-person-team-built-the-open-world-first-then-added-quests-exploration-itself-had-to-feel-like-real-gameplay/ ,
  https://en.prnasia.com/releases/apac/pearl-abyss-unveils-crimson-desert-trailer-commentary-featuring-executive-producer-and-founder-daeil-kim-303421.shtml
* Public showcases: Gamescom ONL 2023 trailer; Gamescom 2024 hands-on boss-rush demo (White Horn,
  Reed Devil, Staglord, Queen Stoneback Crab); TGA 2024 hands-on; GDC 2025 closed-door engine
  session; Summer Game Fest 2025 preview; Gamescom 2025; Gamescom Dev 2025 and CEDEC 2026 talks.
  https://www.techradar.com/gaming/consoles-pc/crimson-desert-shows-off-loads-of-action-in-feature-filled-trailer-at-gamescom-2023 ,
  https://wccftech.com/three-bosses-revealed-in-crimson-desert-demo-at-gamescom-2024/ ,
  https://www.digitaltrends.com/gaming/crimson-desert-hands-on-game-awards-2024/ ,
  https://www.techradar.com/gaming/crimson-desert-preview-sgf-2025 ,
  https://www.invenglobal.com/articles/25231/pearl-abyss-shares-crimson-desert-development-philosophy-at-gamescom-dev
* Development process (CEDEC 2026): three principles "Fast Iteration", "Maintainability",
  "Parallel Workflows"; a **world-first pipeline** (playable world before quests); layered world
  construction (environment layer → content layer → meta-game layer); Houdini for rivers/roads;
  custom XML data formats ("Action Charts", "AI Charts", "Gimmick Charts", "Stage Charts") so
  designers work without engine code; "Phase Switching" (persistent Base Level + changeable
  Gameplay Level) so world states evolve without duplicating maps; iteration cycles cut "from weeks
  to days". Lead quote: "We wanted the open world itself to be the primary source of gameplay."
  https://www.invenglobal.com/articles/24095/pearl-abyss-unveils-crimson-desert-development-process
* Post-launch: the core team moved to a new project after release; Pearl Abyss reported Q1 2026
  revenue up 420% with Crimson Desert 81.3% of revenue.
  https://tribune.com.pk/story/2599900/crimson-desert-development-team-shifts-to-new-project-after-release ,
  https://www.tweaktown.com/news/111561/crimson-desert-made-dollars179-million-revenue-on-4-million-sales-pearl-abyss-confirms/index.html

### 1.3 Engine: BlackSpace (proprietary)

* In-house successor to the Black Desert engine; "manages large-scale worlds through space and
  data streaming based on player position and field of view, rendering vast regions as continuous
  space" (Gamescom Dev 2025, Kim Jin-hwan / Yoon Jin-ho, Game Engine Graphics heads).
  https://www.invenglobal.com/articles/25231/pearl-abyss-shares-crimson-desert-development-philosophy-at-gamescom-dev
* GDC 2025 closed-door demo feature headings (press release wording): "Seamless Open-World
  Exploration", "Detailed Physics and Environmental Interactions", "Dynamic and Realistic Combat",
  "Real-Time Lighting and Atmospheric Effects", "Realistic Water and Fluid Simulation".
  https://www.mynewsdesk.com/uk/swipe-right/pressreleases/pearl-abyss-showcases-blackspace-engine-at-gdc-2025-3376979
* Rendering: real-time ray-traced global illumination (game "relies on ray tracing for lighting
  across the board" per Digital Foundry), stochastic path tracing and real-time colour bleeding
  claimed, unified atmospheric scattering with volumetric clouds, volumetric fog via "fluid and
  froxel raymarching", proxy LODs with hierarchical subdivision for draw distance, light streaming
  through windows, flickering lantern town lighting.
  https://gamesbeat.com/pearl-abyss-unveils-graphics-power-of-the-blackspace-engine-for-crimson-desert/ ,
  https://80.lv/articles/pearl-abyss-demonstrated-crimson-desert-s-blackspace-engine-tech-advancements ,
  https://www.gamereactor.eu/digital-foundry-reviews-the-graphics-in-crimson-desert-1684443/
* Physics: GPU cloth and hair simulation with collision; wind simulation driving trees, grass,
  cloth, hair; destructible structures with "debris generation scaled to force applied"; fire
  propagation; "objects break depending on the force applied, and enemies react naturally to the
  environment"; animation interruption so players can respond mid-animation.
  https://butwhytho.net/2025/03/pearl-abyss-blackspace-engine/ ,
  https://gamesbeat.com/pearl-abyss-unveils-graphics-power-of-the-blackspace-engine-for-crimson-desert/
* Water: **FFT ocean simulation** + **shallow-water simulation** (ripples/flow affected by objects);
  "dynamic wetting" — a volumetric moisture mask wets only submerged parts of body, clothing, armour,
  cape, animal fur.
  https://www.techporn.ph/pearl-abyss-unveils-blackspace-engine-at-gdc-2025/ ,
  https://butwhytho.net/2025/03/pearl-abyss-blackspace-engine/
* Weather: real-time rain/snow/fog/wind, "dynamic temperature changes", NPCs shield themselves from
  rain. https://gamesbeat.com/pearl-abyss-unveils-graphics-power-of-the-blackspace-engine-for-crimson-desert/
* World data numbers (Inven Global, art/engine heads): **20,000+** independent world-management
  units; **500,000** vegetation instances renderable in one scene; **100+** tree species with
  distance-based representation switching; **256 m** sector grid; **5 LOD** levels; GPU-driven
  scatter of medium objects via "placement groups" (density/scale/elevation/slope); small details
  (moss, leaves, pebbles) applied via screen-space buffers; Houdini and Blender wired into the
  pipeline.
  https://www.invenglobal.com/articles/25076/if-you-can-see-it-you-should-be-able-to-go-there-the-secrets-behind-the-art-of-pearl-abyss-crimson-desert
* Upscalers/frame-gen at launch: DLSS 4 (+ later 4.5 variants), FSR 4 "Redstone" incl. Ray
  Regeneration, XeSS (3.0 quality improved in 1.04), frame generation up to 2 generated frames,
  Nvidia Reflex, HDR.
  https://www.techradar.com/gaming/pc-gaming/crimson-desert-will-feature-both-amds-fsr-redstone-and-nvidias-dlss-4-but-we-might-not-even-need-them ,
  https://wccftech.com/crimson-desert-patch-1-04-difficulty-settings-controls-visuals/ ,
  https://esports.gg/guides/crimson-desert/crimson-desert-best-pc-graphics-settings-to-improve-fps
* Awards: Develop:Star Awards 2026 "Best Technical Innovation" (won); China Game Innovation Awards
  2026 "Best International Game" (won); Gamescom Awards 2026 "Most Epic" and "Best PC Game"
  (nominated). https://en.wikipedia.org/wiki/Crimson_Desert

### 1.4 PC specs and file size (official)

| Tier | GPU | CPU | RAM | Target |
|---|---|---|---|---|
| Minimum | GTX 1060 / RX 5500 XT | Ryzen 5 2600X / i5-8500 | 16 GB | 1080p (upscaled from 900p) @ 30 fps |
| Recommended | RTX 2080 / RX 6700 XT | Ryzen 5 5600 / i5-11600K | 16 GB | 1080p60 or 4K30 |
| High | RTX 4070 / RX 7700 XT | Ryzen 5 7600X / i5-12600K | 16 GB | 1440p60 |
| Ultra | RTX 5070 Ti / RX 9070 | Ryzen 7 7700X / i5-13600K | 16 GB | 4K60 |

* OS Windows 10 64-bit 22H2+, DirectX 12, **150 GB SSD required**. Mac: macOS 15+, M2 Pro/M3/M4
  minimum, M3 Pro/M4 Pro/M5 recommended, 150 GB.
  https://www.gameinformer.com/2026/03/10/crimson-desert-specs-for-pc-console-and-other-platforms-revealed ,
  https://store.steampowered.com/app/3321460/Crimson_Desert/
* Actual install: ~93 GB compressed preload → ~121 GB installed; Steam needs ~200 GB free during
  unpack `[single-source]`.
  https://space4games.com/en/games-en/clear-some-space-crimson-deserts-download-size-is-huge/ ,
  https://windowsforum.com/threads/crimson-desert-pc-install-size-and-ssd-guide-plan-200-gb.403877
* Console modes: PS5 Performance 1080p60 (low RT) / Balanced 1280p→4K 40 fps / Quality 1440p→4K
  30 fps (high RT); PS5 Pro Performance 1080p→4K 60 (high RT) / Balanced 1440p→4K 40 / Quality
  native 4K30 (ultra RT); Series X same as PS5; Series S 720p40 or 1080p30 (RT off); ROG Ally X
  720p→1080p 60.
  https://www.gameinformer.com/2026/03/10/crimson-desert-specs-for-pc-console-and-other-platforms-revealed
* Steam Deck: carries the "Verified" badge but reviewers measured sub-30 fps even at lowest
  settings with FSR Ultra Performance; badge widely questioned.
  https://steamdeckhq.com/news/crimson-desert-steam-deck-verified-badge/ ,
  https://www.gfinityesports.com/article/crimson-desert-best-steam-deck-settings
* Digital Foundry: native 4K, ultra, RT ~60 fps on RX 7900 XTX in a pre-release build; RT cost is
  light (~3–4 fps per settings guides) but Ray Reconstruction/Regeneration is expensive.
  https://www.digitaltrends.com/gaming/crimson-desert-delivers-native-4k-ray-tracing-without-upscaling/ ,
  https://www.gamereactor.eu/digital-foundry-reviews-the-graphics-in-crimson-desert-1684443/ ,
  https://esports.gg/guides/crimson-desert/crimson-desert-best-pc-graphics-settings-to-improve-fps
* Ultrawide 21:9/32:9 supported; FOV locked at launch (config-file edits used) `[single-source]`.
  https://evezone.evetech.co.za/launch-radar/crimson-desert-ultrawide-support-day-1

### 1.5 Sales and reception

* Sales: **2 M units in 24 h**; 3 M in first week; 4 M by the Q1 report (≈₩266.5 bn / $179 M
  revenue, PS5 ≈ half of revenue per analyst); **6 M in 3 months**; second-best-selling game of
  2026 at that point. Steam launch peak 239,000 concurrent (PC Gamer); all-time peak ~275,859
  (SteamPulse).
  https://en.wikipedia.org/wiki/Crimson_Desert ,
  https://www.tweaktown.com/news/111561/crimson-desert-made-dollars179-million-revenue-on-4-million-sales-pearl-abyss-confirms/index.html ,
  https://www.gamesradar.com/games/open-world/crimson-desert-has-made-around-usd200-million-on-its-4-million-copies-sold-analyst-estimates-and-playstation-5-sales-account-for-almost-half-of-that/ ,
  https://www.mmorpg.com/news/crimson-desert-sells-6-million-copies-pearl-abyss-shares-june-september-roadmap-and-confirms-dlc-plans-2000138277 ,
  https://www.pcgamer.com/games/action/crimson-desert-launches-to-239-000-players-on-steam-but-mixed-reviews-and-its-mostly-because-of-how-dense-and-cryptic-the-whole-thing-is/ ,
  https://www.steampulse.org/game/3321460
* Critics: Metacritic **77 (PC)** / **78 (PS5)**; OpenCritic top-critic average 79–81, **75 %
  recommend**. Metacritic user score ~8.8 `[uncertain]`. Scores span 4.5/10 to 9.5/10. Pearl Abyss
  stock fell on review day because investors expected 80+.
  https://en.wikipedia.org/wiki/Crimson_Desert ,
  https://opencritic.com/game/19373/crimson-desert ,
  https://www.windowscentral.com/gaming/crimson-desert-reviews-and-metacritic-scores-are-in-ahead-of-launch-heres-what-everyones-saying ,
  https://www.gamespot.com/articles/crimson-desert-devs-stock-plunges-following-the-games-reviews/1100-6538890/
* Individual scores: IGN 6/10 (Travis Northup, 110+ h, "review in progress" then final); PC Gamer
  80 (Mollie Taylor); Eurogamer 3/5; GameSpot 7 (Richard Wakeling); Game Informer 7 (Hayes
  Madsen, 100 h); Destructoid 8.5; PCGamesN 6; Push Square 6 (PS5); TheGamer 4/5; GamesRadar 4/5;
  Game Rant 8; Vice 5/5; Forbes 9.5; Critical Hits 4.5.
  https://www.thegamer.com/crimson-desert-review-round-up/ ,
  https://www.pushsquare.com/features/round-up-crimson-desert-reviews-are-a-major-disappointment ,
  https://kotaku.com/crimson-desert-metacritic-scores-review-roundup-2000680110
* Steam user reviews: "Very Positive" 86 % of ~62.9 k overall; "Mixed" at launch, "Mostly
  Positive" 76 % recent (Oct 2026).
  https://store.steampowered.com/app/3321460/Crimson_Desert/ ,
  https://www.pcgamer.com/games/action/crimson-desert-launches-to-239-000-players-on-steam-but-mixed-reviews-and-its-mostly-because-of-how-dense-and-cryptic-the-whole-thing-is/
* Controversies: no Intel Arc support at launch (refunds suggested, later "unoptimized" patch);
  AI-generated paintings/signs found in-world, Pearl Abyss apologised and pledged replacement; no
  AI voice acting was used.
  https://en.wikipedia.org/wiki/Crimson_Desert ,
  https://thegameswiki.com/crimson-desert/wiki/voice-acting

---

## 2. World

### 2.1 Setting and regions

* Continent **Pywel**, high fantasy that drifts into sci-fi (mechs, clockwork, "Tesla Ruins").
  https://en.wikipedia.org/wiki/Crimson_Desert , https://beebom.com/crimson-desert-map/
* Five surface regions + one alternate dimension:
  * **Hernand** — starting region; green plains, forests, big cities, European high-fantasy
    (Hernand Castle, The Eldertree, Calphade territory).
  * **Pailune** — northern alpine; snow, blizzards, Frozen Soul Mountain, Dragon Ridge; Greymanes'
    homeland.
  * **Demeniss** — Pywel's capital/political centre; military installations, sieges.
  * **Delesyia** — most technologically advanced; robots, automatons, aerial research base.
  * **The Crimson Desert** — lawless red-sand wasteland; brigands, Bonepit arena.
  * **The Abyss** — alternate dimension reached via Abyss Gateways; Library of Providence; final
    chapter.
  https://beebom.com/crimson-desert-map/ , https://www.powerpyx.com/crimson-desert-full-world-map/ ,
  https://thegameswiki.com/crimson-desert/wiki/hernand-region-guide
* Fixed climates per region, no seasons; desert gets sandstorms never snow, alpine gets blizzards,
  coasts cycle fog/rain `[single-source]`.
  https://steamcommunity.com/sharedfiles/filedetails/?id=3689066736

### 2.2 Map size and density

* Surface **~90 km² (9,500 m × 9,500 m)**, ~100 km² including underground/sky layers; press
  comparisons: ~2× Skyrim, larger than RDR2.
  https://www.powerpyx.com/crimson-desert-full-world-map/ , https://beebom.com/crimson-desert-map/ ,
  https://www.g2a.com/news/features/crimson-desert-map-size-explained-how-big-is-the-world-of-pywel/
* **780+ named landmarks**: Hernand 240, Demeniss 179, Crimson Desert 136, Delesyia 115, Pailune
  111 `[single-source]`. https://mapmaster.io/games/crimson-desert/guides/Landmark
* **100+ caves**, 37 Ancient Ruins, 40 Abyss Restoration sky-island puzzles, 8 Spires, numerous
  Sanctums, bandit camps, blockades, quarries, outposts (see §5).
  https://www.powerpyx.com/crimson-desert-full-world-map/ ,
  https://thegameswiki.com/crimson-desert/wiki/ancient-ruins-and-their-puzzles ,
  https://vulkk.com/2026/06/14/abyss-challenges-list-in-crimson-desert-conqueror-of-the-abyss-achievement/
* Three vertical layers: surface, underground (caves/sanctums), sky (floating "Abyss islands" for
  puzzle challenges, compared to TotK).
  https://www.powerpyx.com/crimson-desert-full-world-map/
* Map fog-of-war: grey fog cleared by ringing **8 Bells** or exploring; main quests orange,
  faction/side quests silver; patch 1.04 added map filters/search; 1.18 added a better fog filter.
  https://www.powerpyx.com/crimson-desert-full-world-map/ ,
  https://thegameswiki.com/crimson-desert/wiki/quest-system ,
  https://vulkk.com/2026/08/15/crimson-desert-update-1-18-patch-notes-better-map-fog-filter-controls-and-knowledge-improvements/
* Seamless: no loading between regions; "you can, quite literally, see every inch of it from any
  high point" (Game Informer). One guide claims a "region-based structure rather than seamless"
  — contradicted by reviews `[uncertain]`.
  https://gameinformer.com/review/crimson-desert/open-world-overload ,
  https://pyweldb.com/guides/story-chapters/

### 2.3 Day/night, weather, simulation

* Persistent calendar: HUD shows day number, weekday, time, temperature; **1 real minute ≈ 12
  in-game minutes** (≈2 h full cycle) `[single-source]`; sleep in any unoccupied bed or at a
  bonfire to skip 3/6/12 h (with cooldown); sleeping clears fatigue.
  https://steamcommunity.com/sharedfiles/filedetails/?id=3689066736 ,
  https://game8.co/games/Crimson-Desert/archives/588561 ,
  https://www.ludo.guide/guide/crimson-desert/game-mechanics-and-tips/how-to-skip-time-sleep
* Weather computed at runtime from biome, temperature, elevation, wind, time (not scripted);
  cutscenes rendered in real time so they inherit current weather/time.
  https://thegameswiki.com/crimson-desert/wiki/weather-system ,
  https://thegameswiki.com/crimson-desert/wiki/npc-daily-routines
* Weather gameplay: wind changes glide distance/stamina; fog shrinks enemy detection range (stealth
  aid); some enemies spawn only at night; rain/snow render in boss fights.
  https://steamcommunity.com/sharedfiles/filedetails/?id=3689066736
* Temperature/survival: gauge beside minimap (blue cold / red hot); away from neutral → stamina
  regen drops and actions cost more; countered by cloaks (e.g. Reindeer Cloak Ice Resistance 5),
  Frostward Abyss Gear, consumables (beer, honey tea, clear soup), standing by campfires; one guide
  also cites hydration/shelter `[uncertain]`. No hunger system in base game (community mod exists).
  https://steamcommunity.com/sharedfiles/filedetails/?id=3689066736 ,
  https://crimsondesert.club/guides/world/survival-mechanics ,
  https://www.nexusmods.com/crimsondesert/mods/806?tab=posts
* NPC schedules: villagers wake, work (farming, fishing, guard duty), sleep on the day-night cycle;
  shops keyed to routines; players "love watching NPCs work in real time" (GameSpot). Pearl Abyss:
  "all villagers going about their daily lives and every NPC, event, and side quest in the game were
  created for a purpose."
  https://www.gamespot.com/articles/crimson-desert-players-love-watching-npcs-work-in-real-time/1100-6539161/ ,
  https://thegameswiki.com/crimson-desert/wiki/npc-daily-routines
* Criticised limit: towns full of NPCs with only a "generic greeting" and "no incidental story"
  (PCGamesN). https://www.pcgamesn.com/crimson-desert/review

### 2.4 Traversal

* **Climbing**: walk into almost any rock face and jump; limited by stamina not permission; falling
  when stamina hits zero; food can be eaten mid-climb/swim/glide from the consumable wheel.
  https://www.slashskill.com/crimson-desert-tips-and-tricks-hidden-mechanics-the-game-never-explains/ ,
  https://crimsondesert.wikirealm.com/guides/traversal-and-mounts/
* **Jumping**: double jump early; aerial Force Palm (R3) up to three times mid-air to vault cliffs.
  https://www.slashskill.com/crimson-desert-tips-and-tricks-hidden-mechanics-the-game-never-explains/
* **Gliding**: "Crow-Wing" glider ("Flight" skill) from the "Woman in White" witch quest in
  Hernand; always descends; hold to speed up at extra stamina; duration upgradable; Oongka uses a
  rocket pack instead.
  https://crimsondesert.wikirealm.com/guides/traversal-and-mounts/ ,
  https://steamcommunity.com/sharedfiles/filedetails/?id=3693402643
* **Axiom Force** (grapple/telekinesis from the Axiom Bracelet, Abyss questline): spectral claw
  that latches to anchors/objects/enemies to push, pull, rotate; upgrades "Aerial Maneuver" and
  "Aerial Swing" make a "physics slingshot"/"Spider-Man-style" traversal (Vice).
  https://crimsondesert.fandom.com/wiki/Axiom_Force ,
  https://www.vice.com/en/article/crimson-desert-review-the-most-ambitious-open-world-game-since-red-dead-redemption-2/
* **Swimming**: stamina-based, drowning on empty; no diving in base game (DLC adds underwater
  exploration). https://thegameswiki.com/crimson-desert/wiki/how-to-increase-stamina ,
  https://en.wikipedia.org/wiki/Crimson_Desert
* **Mounts**: horse from the start (wild horses tamed via minigame, registered, summonable); bears,
  wyverns (≈15 min flight on ≈50 min cooldown `[single-source]`), dragons (Abyssal Dragon from the
  Abyss storyline), war mechs (crafted from blueprints/keys), "29 mounts across 8 categories"
  `[single-source]`; horses have trust 1–5, randomised stats, 4 gear slots.
  https://crimsondesertwiki.net/articles/mount-guide-all-29-mounts-combat-capabilities-and-how-to-get-them ,
  https://thegameswiki.com/crimson-desert/wiki/horse-guide ,
  https://crimsondesert.wikirealm.com/guides/traversal-and-mounts/
* **Vehicles**: wagons, skiffs/boats (no stamina drain on water), hot-air balloons (camp
  expansion unlock); caravan/wagon hijacking exists.
  https://thegameswiki.com/crimson-desert/wiki/skiffs-and-wagons ,
  https://www.method.gg/crimson-desert/crimson-desert-greymane-camp-guide-how-to-upgrade-and-manage-your-camp ,
  https://www.pushsquare.com/reviews/ps5/crimson-desert
* **Fast travel** must be earned: stand on an **Abyss Nexus** pressure plate to activate (blue
  icon); **Abyss Cressets** (~37) are puzzle-gated at Ancient Ruins and also award an Artifact; no
  fast travel while mounted, in combat or swimming; "Blinding Flash" (hold lantern + attack) and
  "Guiding Light" (sword drawn) reveal nearby Abyss points.
  https://powerupgaming.co.uk/2026/03/24/crimson-desert-air-travel-fast-travel-and-exploration-tricks/ ,
  https://www.pywel.app/en/guides/abyss-system/ ,
  https://crimsondesert.wikirealm.com/guides/traversal-and-mounts/
* Dev-stated exploration contract: "If you can see it, you should be able to go there"; "Exploration
  takes priority over beauty" (Ahn Geun-tae, Art Level head).
  https://www.invenglobal.com/articles/25076/if-you-can-see-it-you-should-be-able-to-go-there-the-secrets-behind-the-art-of-pearl-abyss-crimson-desert

### 2.5 Exploration rewards and environmental interaction

* Rewards: Abyss Artifacts (= skill points) from puzzles, Cressets, sky islands, liberation, hidden
  bosses that unlock abilities; Contribution XP; treasure chests; Memory Fragments; Knowledge
  entries; mounts/pets; recipes.
  https://www.pywel.app/en/guides/abyss-system/ ,
  https://vulkk.com/2026/03/31/how-to-liberate-and-clear-locations-in-crimson-desert/ ,
  https://gameinformer.com/review/crimson-desert/open-world-overload
* Physics/destruction: vegetation reacts to movement and can be destroyed; walls crumble, trees
  fall, watchtowers level, carts splinter; explosive barrels; enemies thrown into columns/off
  cliffs; **fire propagates** through wood and dry grass and burning enemies ignite terrain;
  "destruction is systemic rather than hand-authored".
  https://thegameswiki.com/crimson-desert/wiki/environmental-combat-and-physics ,
  https://gamerblurb.com/articles/crimson-desert-how-to-destroy-buildings-wagons-totems ,
  https://gamingbolt.com/crimson-desert-30-brilliant-details-that-elevate-the-entire-experience
* Objects are data-driven "gimmicks" with states; a player-found aerial-flight bug was kept as a
  feature. https://www.invenglobal.com/articles/24095/pearl-abyss-unveils-crimson-desert-development-process

---

## 3. Player character

### 3.1 Playable characters

* **Kliff Macduff** (default): sword + shield + bow, wrestling moves, elemental magic via the Axiom
  Bracelet; "evasion/hybrid" style. Voiced by Alec Newman.
  https://www.gamesradar.com/games/rpg/crimson-desert-characters/ ,
  https://gfuel.com/blogs/news/crimson-desert-all-main-voice-actors
* **Damiane** (unlocked Chapter 3 when the camp opens): claymore or rapier, buckler, pistol/musket,
  spells; precise dodging; "aggression". **Oongka** (unlocked ~Chapter 8): giant axe, wrist cannon,
  grabs/slams, rocket-pack flight; "AoE". Switch any time from a character wheel ("similar to GTA
  5"). Forced segments with under-levelled alts criticised (Destructoid).
  https://mein-mmo.de/en/crimson-desert-the-3-playable-characters-and-their-characteristics-which-one-suits-you,1551179/ ,
  https://beebom.com/crimson-desert-characters/ ,
  https://www.destructoid.com/reviews/crimson-desert-review/
* No character creator; cosmetic dye station and outfits; option to hide back weapons (added
  post-launch). https://steamcommunity.com/app/3321460/discussions/0/761806896608981679/ ,
  https://www.shacknews.com/article/148665/crimson-desert-boss-fight-rematch

### 3.2 Stats and progression (no XP levels)

* Three core stats: **Health, Stamina, Spirit**. Health can be raised 18 times, Stamina 16 (early
  guide) / Stamina Level 0→10 then to 14 via Red Seaweed Research at Urdavah `[uncertain, sources
  differ]`.
  https://thegameswiki.com/crimson-desert/wiki/early-progression-guide ,
  https://thegameswiki.com/crimson-desert/wiki/how-to-increase-stamina
* **Abyss Artifacts** replace levelling: each regular Artifact = 1 skill point (spend on skills or
  on Health/Stamina/Spirit). Sources: an uncapped enemy-kill gauge, story quests, boss drops,
  puzzles, Cressets, sky islands, liberation, vendors (~28.5 silver each). **141 Sealed Artifacts**
  at shrines need a hidden challenge completed (e.g., "kill N enemies with weapon type X");
  **Faded Artifacts** reset all skill points.
  https://steamcommunity.com/sharedfiles/filedetails/?id=3698407367 ,
  https://www.pywel.app/en/guides/abyss-system/
* 2.0 rework: **Abyss Links** — when one character spends Artifacts, the others receive equivalent
  Links; per-character resets; all spent resources refunded on upgrade; 8 new Kliff skills gated by
  story progress.
  https://www.gfinityesports.com/article/crimson-desert-update-20000-patch-notes-new-story-content-fixes ,
  https://gameriv.com/crimson-desert-enhanced-update-adds-new-story-content-voice-languages-skills-and-more/
* Skill trees: Kliff 80+ skills, Damiane ~70, Oongka ~70; branches for melee combos, defence,
  mobility, grappling, elemental, life skills; unlock via Artifacts, story beats, or **"Watch and
  Learn"** (observing an enemy's move in combat fills a bar that unlocks it — e.g., learn
  "Clothesline" by being hit by it).
  https://steamcommunity.com/sharedfiles/filedetails/?id=3693402643 ,
  https://thegameswiki.com/crimson-desert/wiki/skill-tree-guide ,
  https://vulkk.com/2026/03/31/how-to-liberate-and-clear-locations-in-crimson-desert/
* **Knowledge** codex: 2,921 entries (people, places, creatures, recipes, bosses, factions),
  auto-tracked; some skills/upgrades only come from discovery; patch 1.18 added quest knowledge.
  https://thegameswiki.com/crimson-desert/wiki/knowledge ,
  https://vulkk.com/2026/08/15/crimson-desert-update-1-18-patch-notes-better-map-fog-filter-controls-and-knowledge-improvements/

### 3.3 Equipment, inventory, upgrades

* Weapon categories: one-handed (sword, dagger, rapier, hammer/mace + shield off-hand), two-handed
  (greatsword, axe, spear, staff), ranged (bow, pistol, rifle, hand cannon), unarmed/grapple, mount
  weapons. Databases list **535+ weapons**; 58 "named" examples in one guide `[uncertain]`.
  https://crimsondesertgame.wiki.fextralife.com/Weapons , https://beebom.com/crimson-desert-weapons/ ,
  https://crimsondb.gg/weapons?type=spear
* Quick Swap skill lets Kliff carry a secondary weapon; other characters swap on demand.
  https://www.thegamer.com/crimson-desert-best-combat-tips-tricks-fighting-parrying/
* Inventory is **slot-based, not weight-based**: 50 slots → cap 240 via Small (+1, 50 copper),
  Medium (+3, quests) and Large (+5, story) bags; every weapon/armour piece = 1 slot, stackables to
  50; camp Supply Chest 230 slots; patch 1.04 added 1,000-slot gatherables/collectibles chests,
  coolers (40/330) and wardrobes (100 each, up to 1,000). Inventory friction is the most repeated
  review complaint.
  https://www.powerpyx.com/crimson-desert-how-to-increase-inventory-size/ ,
  https://crimsondeserthq.com/blog/inventory-management-guide ,
  https://wccftech.com/crimson-desert-patch-1-04-difficulty-settings-controls-visuals/ ,
  https://www.pushsquare.com/reviews/ps5/crimson-desert
* **Refinement**: every weapon/armour/shield/jewellery has 10 levels, ~+8 % base stats per level;
  levels 1–4 use ores/timber/hides/bones, 5+ also consume Abyss Artifacts; materials only, no
  silver; blacksmiths in towns and (when upgraded) at camp. Patch 1.04 reduced attack/defence from
  reinforcement.
  https://www.keengamer.com/articles/guides/crimson-desert-refinement-guide-how-to-upgrade-weapons-and-armor/ ,
  https://game8.co/games/Crimson-Desert/archives/587627
* **Abyss Gear / Cores**: socketed passives (Swift, Destruction, Fortification families; unique boss
  cores like "Crow's Pursuit"); socket count scales with refinement tier (Tier 5 = up to 5).
  https://www.pywel.app/en/guides/abyss-system/
* Armour: plate sets, cloaks with elemental resistance, faction/contribution-shop sets; lighter
  armour helps in heat `[single-source]`. https://steamcommunity.com/sharedfiles/filedetails/?id=3689066736
* Consumables: Palmar Pill (instant revive at 30 % HP), stamina/HP/Spirit foods, potions
  (Attack Speed +30 %, Spirit restore) from the camp Alchemy Lab.
  https://thegameswiki.com/crimson-desert/wiki/death-penalty ,
  https://crimsondesertwiki.net/articles/greymane-camp-guide-all-upgrades-life-skills-and-base-building

### 3.4 Crafting, cooking, life skills

* Cooking at any bonfire/campfire/cauldron: throw ingredients in; valid combos yield a dish; three
  successes memorise the recipe; **40+ recipes**, four quality tiers (Modest/Basic/Filling/Hearty);
  dishes restore HP/Spirit/Stamina and give resistances/combat windows; the primary healing source.
  https://www.keengamer.com/articles/guides/crimson-desert-cooking-alchemy-and-healing-complete-guide/ ,
  https://consolepulse.com/multiplatform/crimson-desert/guides/crimson-desert-all-cooking-recipes-best-food-buffs
* Life skills: alchemy, cooking, farming, ranching, mining, herbalism/gathering, logging, fishing,
  hunting, woodwork/stonework workstations, looms (textiles); proficiency speeds actions and yields
  bonus materials.
  https://crimsondesertwiki.net/articles/greymane-camp-guide-all-upgrades-life-skills-and-base-building ,
  https://games.gg/crimson-desert/guides/crimson-desert-greymane-camp-guide/
* Camp resource ledger has five categories: Armament, Peter (minerals), Wood, Food, Silver.
  https://games.gg/crimson-desert/guides/crimson-desert-greymane-camp-guide/

### 3.5 Camp, housing, base building

* **Greymane Camp at Howling Hill** (Chapter 3 "Homestead"): starts as tents; grows via recruiting
  scattered Greymanes and spending resources/Contribution into blacksmith, forge (L3 crafts top-tier
  gear), infirmary (passive heal), alchemy lab, cooking station, provisions tent, dye station,
  mission board, merchants, farm, ranch, trading post, personal house, wagons and balloons.
  https://crimsondesertwiki.net/articles/greymane-camp-guide-all-upgrades-life-skills-and-base-building ,
  https://www.method.gg/crimson-desert/crimson-desert-greymane-camp-guide-how-to-upgrade-and-manage-your-camp
* **Dispatch missions**: send recruited comrades on timed missions (solo and "Duo" pairings) for
  loot/Contribution; camp morale affects companion performance; Supply Chest collects returns.
  https://crimsondesertwiki.net/articles/greymane-camp-guide-all-upgrades-life-skills-and-base-building ,
  https://games.gg/crimson-desert/guides/crimson-desert-greymane-camp-guide/
* **Housing**: furniture placement, multiple house layouts, bulk retrieval, outdoor overhaul (patch
  1.04); DLC expands housing.
  https://gamertagmythras.com/blog/crimson-desert/crimson-desert-housing-guide ,
  https://wccftech.com/crimson-desert-patch-1-04-difficulty-settings-controls-visuals/
* **Farming/ranching**: crops need watering/fertiliser; livestock capacity/breeding/feed; outputs
  feed cooking and crafting. https://www.whisperofthehouse.com/crimson-desert/farming-ranching-guide

### 3.6 Pets and companions

* Pets tamed by raising trust 0→100 (pet, feed): dogs, cats (5 new types in 1.04), birds (Sotdae of
  Bond), legendary birds, Abyss creatures, egg pets; pets auto-loot; **Sigil of Valor** makes pets
  fight; **Sigil of Bonding** keeps shoulder cats on during climbing/gliding/riding/combat; pet
  accessory slot; renaming.
  https://games.gg/crimson-desert/guides/crimson-desert-pets-guide/ ,
  https://www.gfinityesports.com/article/how-to-get-the-sigil-of-valor-in-crimson-desert-and-unlock-pet-combat ,
  https://wccftech.com/crimson-desert-patch-1-04-difficulty-settings-controls-visuals/
* Human companions mostly act via camp/dispatch; allied armies fight alongside in sieges; some
  story segments pair Kliff with Damiane/Oongka. https://thegameswiki.com/crimson-desert/wiki/siege-battles

---

## 4. Combat

### 4.1 Core loop

* Inputs (pad): light RB/R1; heavy RT+RB; parry/guard LB/L1; dodge B/Circle; draw bow LT; Force
  Palm R3; lantern = Blinding Flash. Light attacks cost no stamina; sprint, dodge, block, heavy
  attacks and some skills drain it.
  https://www.thegamer.com/crimson-desert-best-combat-tips-tricks-fighting-parrying/ ,
  https://carrylord.com/crimson-desert/combat-guide/
* **Parry**: hold guard and release as the hit lands; green flash; enemy staggers; counter window
  deals ~2.5× damage; refills Stamina and Spirit. **Red-glow attacks are unblockable** — dodge.
  https://crimsondeserthq.com/blog/combat-guide-parry-dodge ,
  https://bossdown.com/guides/crimson-desert-combat-guide/
* **Perfect dodge / backstep** (Keen Senses Lv2) also refunds Stamina/Spirit; "Counter" skill =
  attack right before being hit to interrupt; patch 1.04 added double-tap-or-hold dodge input.
  https://crimsondeserthq.com/blog/combat-guide-parry-dodge ,
  https://steamcommunity.com/sharedfiles/filedetails/?id=3693402643 ,
  https://wccftech.com/crimson-desert-patch-1-04-difficulty-settings-controls-visuals/
* **Grapple/wrestling**: grab staggered or low-HP enemies; sprint-grab pins; wall/cliff slam; rear
  **Suplex** with AoE knockdown; Lariat, Throw, Back Hang (airborne latch), Clothesline,
  Chokeslam/Angle Slam shown in showcases; moves mo-capped from real wrestlers/Taekwondo athletes.
  https://crimsondeserthq.com/blog/combat-guide-parry-dodge ,
  https://www.windowscentral.com/gaming/crimson-deserts-2nd-gameplay-showcase-highlights-combat-and-player-progression-i-can-chokeslam-and-angle-slam-enemies-in-this-game-sign-me-up-now ,
  https://mmoculture.com/2020/12/crimson-desert-pearl-abyss-interview-clarifies-single-player-and-multi-player-questions-and-more/
* **Resources**: Health; Stamina; **Spirit** (fuel for active skills/elements; restored by parries,
  food, potions). https://steamcommunity.com/sharedfiles/filedetails/?id=3693402643
* **Stagger/posture**: enemies stagger on parry, blunt weapons stagger easily; warlords/bosses have
  super-armour and multiple HP bars; 1.04 removed boss immunity during big attacks and tuned
  counter/escape frequency.
  https://beebom.com/crimson-desert-weapons/ , https://dexora.gg/crimson-desert/guides/enemy-weaknesses ,
  https://wccftech.com/crimson-desert-patch-1-04-difficulty-settings-controls-visuals/
* **Executions/finishers** ("Blinded Flash Finisher" etc.) — Game Informer found execution
  animations exhausting against 40+ enemies.
  https://gameinformer.com/review/crimson-desert/open-world-overload
* Dev framing: "combo-based controls, fluid transitions, and player-driven skill expression"; "not
  designed as a Souls-like"; previewers call it "more like Street Fighter than Soulslike".
  https://www.gamereactor.eu/we-talk-difficulty-scale-and-inspirations-in-crimson-desert-with-pearl-abyss-1674833/ ,
  https://screenrant.com/crimson-desert-gamescom-2024-hands-on-action-rpg/

### 4.2 Weapon movesets and magic

* Movesets per category: swords (sweeps, dual-wield), daggers (weak points/assassination), rapiers
  (fast thrusts), hammers (stagger), greatswords (wide multi-hit), axes (shorter, heavy), spears
  (reach), staffs (magic), bows (full archery branch), pistols (quick shots), rifles (kick), hand
  cannons (heavy), shields (parry/block/counter), unarmed "monk-style" punches/kicks/grapples.
  https://beebom.com/crimson-desert-weapons/
* Skill examples: Forward Slash, Turning Slash, Evasive Roll, Double Jump, Flight, Force Palm
  (knockback + defence down), Nature's Snare, Blinding Flash, Flame Strike/Quake, Frost Mantle,
  Lightning Surge/Pulse, Storm Veil/Howl.
  https://steamcommunity.com/sharedfiles/filedetails/?id=3693402643 ,
  https://www.thegamer.com/crimson-desert-best-combat-tips-tricks-fighting-parrying/
* **Axiom Bracelet** = Kliff's only magic source: dial UI switches **fire / lightning / ice**
  imbues mid-fight; wind appears in skills/traversal rather than imbue. Fire = burn DoT that spreads
  to terrain; lightning staggers elites and chains through groups; ice slows heavy beasts. 1.04
  raised elemental ailment damage.
  https://crimsondesert.fandom.com/wiki/Axiom_Bracelet ,
  https://thegameswiki.com/crimson-desert/wiki/fire-element ,
  https://dexora.gg/crimson-desert/guides/enemy-weaknesses
* Ranged: bow draw/release; signal/whistling/explosive arrows in sieges; mounted ranged lock-on
  fixed in 1.18. https://thegameswiki.com/crimson-desert/wiki/siege-battles ,
  https://gameriv.com/crimson-desert-patch-1-18-00-all-major-changes-fixes-and-new-features/
* Mounted combat: horses fight at higher trust; bears bite/charge; dragons breathe fire; mechs fire
  missiles; some bosses are designed around mounting/climbing them.
  https://beebom.com/crimson-desert-weapons/ ,
  https://www.oslink.io/blog/guide/crimson-desert-boss-guide.html

### 4.3 Camera, lock-on, hit reactions

* Lock-on was the most-cited combat flaw ("killer combat (pesky lock-on aside)" — Eurogamer; soft
  lock picks adds over bosses, hard lock loses leaping targets). Patch 1.18 added Manual /
  Semi-auto / Auto lock-on camera rotation modes.
  https://www.pushsquare.com/features/round-up-crimson-desert-reviews-are-a-major-disappointment ,
  https://gamingbolt.com/crimson-desert-is-better-than-before-yet-these-15-issues-still-hurt-it ,
  https://vulkk.com/2026/08/15/crimson-desert-update-1-18-patch-notes-better-map-fog-filter-controls-and-knowledge-improvements/
* Controls described as "wonky"/"on skates"; players "careening off the side of a cliff" (Game
  Informer, PCGamesN); 1.04 added new control presets ("Classic" kept) and reworked vault.
  https://gameinformer.com/review/crimson-desert/open-world-overload ,
  https://www.pcgamesn.com/crimson-desert/review
* Physics-driven hit reactions: enemies take environmental contact damage (pillars/trees — reduced in
  1.04), ragdoll into props, fire/ice interact with terrain; GameSpot: "might have the most realistic
  in-game physics I've ever seen" `[pre-release]`.
  https://butwhytho.net/2025/03/pearl-abyss-blackspace-engine/ ,
  https://www.gamespot.com/articles/crimson-desert-might-have-the-most-realistic-in-game-physics-ive-ever-seen/1100-6530297/

### 4.4 Enemy archetypes

* Humans: bandits/footsoldiers (2–3 hit combos, shield bash), elite soldiers/mercenaries (weapon
  swaps, parry you), warlords (mini-bosses, multi HP bars, unblockables), gun-wielding outpost
  troops, Black Bear soldiers, siege engines.
  https://dexora.gg/crimson-desert/guides/enemy-weaknesses ,
  https://vulkk.com/2026/03/31/how-to-liberate-and-clear-locations-in-crimson-desert/
* Beasts: pack hunters (wolves, hyenas with an alpha), heavy beasts (bears, boars, trolls), colossal
  creatures (weak points, mount during staggers). Goblins: scouts (hit-and-run), warriors/siege units
  (shield formations). Undead/magical: skeletons/revenants (resurrect unless burned), sorcerers
  (summons, mines), urn soldiers, specters; harpies; mechanical automatons, clockwork constructs,
  mech dragons. Community criticism that variety is thin for the map size.
  https://dexora.gg/crimson-desert/guides/enemy-weaknesses ,
  https://thegameswiki.com/crimson-desert/wiki/enemies-and-threats ,
  https://steamcommunity.com/app/3321460/discussions/0/805720464938001556/

### 4.5 Bosses (named)

* Count: "75+" (Game Rant) / **80** (crimsondb) incl. story, faction, exploration, secret, and three
  "Overwhelming Beings" super-bosses. https://gamerant.com/crimson-desert-all-boss-fights-locations-quests/ ,
  https://crimsondb.gg/bosses
* Hernand: Marni's Excavatron, Saigord the Staglord (3 HP bars, fury phase), Queen Spider, Cubewalker
  Lithus, Runewalker Vordis, The Crimson Nightmare, Giath, Sizlek the Insatiable, Sir Catfish, Hemon
  Beindel, Gwen Kraber, Cassius Morten, Reed Devil (illusion clones, 3 phases), Kearush the Slayer,
  Kailok the Hornsplitter, Walter Lanford.
* Pailune: Frostwalker Tervis, Karanda (harpy), White Horn Shepherd of Souls (yeti; climb its back,
  fire weakness, ice breath P2), Mudwalker Lutemir, Bloodwalker Crussis, White Bearclaw, White Bear
  of the High Mountains, Tarandus the Ashen, Myurdin, Ludvig, Titan, Black Bear Siege Engine, Moren
  the Mistwood Hunter.
* Demeniss: Gabriel Caliburn, Skull Knight, Stonewalker Antiquum, Trukan the Ascended, Gregor the
  Halberd of Carnage, Tristan the Flame Knight, Bradie Gu, The Raging Tempest, Mazuul the Dark
  Justiciar, Fortain the Cursed Knight, The Crimson Warden, Keglord Garnier Mk. XXIII, Lucian
  Bastier, The Lunar Reapers, Hexe Marie, Draven the Crowcaller.
* Delesyia: Marni's Clockwork Mantis, Thunder Tank, Storm Crusher, Queen Stoneback Crab (climb the
  shell, destroy silver weak points), Clockwork White Horn, Balthazar the Wyvernflame, Mechanicus,
  Ironwing REN-X, Golden Star (mechanical dragon).
* Crimson Desert: Queen Bismuth Oreback Crab, Muskan, The Masked Liberator, Samara the Sandwatcher,
  Merrick Knight of Fortune, Gristle the Sandfang Marauder, Ogre, Ravok of the Savage Fangs.
* Pywel-wide/secret: Kutum, Antumbra's Spear/Sword/Staff, Priscus/Praevus/Primus the Ancient,
  Corrupted Caliburn, Myurdin Avatar of Umbra, **Umbra** (final), Aeserion the Great Serpent.
  Overwhelming Beings: The Forgotten General, Beloth the Darksworn, Ator Archon of Antumbra.
  https://gamerant.com/crimson-desert-all-boss-fights-locations-quests/ ,
  https://www.oslink.io/blog/guide/crimson-desert-boss-guide.html ,
  https://crimsondb.gg/bosses
* Boss design critique: "totally unbalanced" spikes, tiny arenas vs huge AoEs, gimmick kills, no
  indication of enemy strength (Destructoid, Game Informer, Push Square); praised as Shadow of the
  Colossus-style set pieces in previews.
  https://www.destructoid.com/reviews/crimson-desert-review/ ,
  https://gameinformer.com/review/crimson-desert/open-world-overload ,
  https://screenrant.com/crimson-desert-gamescom-2024-hands-on-action-rpg/
* Boss rematches and "re-blockade" (enemies retake liberated areas) were announced in development
  April 2026 `[uncertain whether shipped]`. https://www.shacknews.com/article/148665/crimson-desert-boss-fight-rematch

### 4.6 Difficulty

* Launched with **no difficulty modes** (dev: tune via gear, consumables, abilities). Patch
  **1.04 (23 Apr 2026)** added Easy / Normal / Hard: Easy widens parry/dodge windows and lowers enemy
  stats; Hard tightens windows, cuts roll i-frames, delays food effects to end of animation, and
  gives some bosses new patterns.
  https://www.gamereactor.eu/we-talk-difficulty-scale-and-inspirations-in-crimson-desert-with-pearl-abyss-1674833/ ,
  https://www.vice.com/en/article/crimson-desert-patch-notes-1-04-just-dropped-with-new-difficulty-modes-and-major-changes/ ,
  https://www.keengamer.com/articles/guides/crimson-desert-difficulty-settings-guide-easy-normal-and-hard/

---

## 5. Systems

### 5.1 Quests

* Structure: Prologue "Dead of Night" + **12 chapters** + Epilogue "Journey's End"; ~168 main
  quests `[single-source]`; Chapter 1 "The First Encounter", Ch. 6 "Cracks in the Shield" (Calphade
  siege chapter), Ch. 8 "Blood Coronation", Ch. 12 "The Abyss".
  https://pyweldb.com/guides/story-chapters/ ,
  https://thegameswiki.com/crimson-desert/wiki/chapter-6-cracks-in-the-shield
* Types: main; faction questlines (186–280+ depending on counting); commission/notice boards per
  region; bounties; liberation; investigation/contradiction quests (pick the contradicting
  statement); research quests; daily-life quests; constellation activities; organically discovered
  "rumour" quests; dispatch missions. Pearl Abyss's own figure: **430 "adventures"** on top of the
  main story; 63 side quests in the first town alone.
  https://thegameswiki.com/crimson-desert/wiki/quest-system ,
  https://crimsondb.gg/faction-quests ,
  https://crimsondesertwiki.net/articles/investigation-contradiction-answers-guide ,
  https://beebom.com/how-long-to-beat-crimson-desert/
* Quest-design reception: "following a checklist" (GameSpot); padded fetch loops like a 15-minute
  dye quest (Game Informer); "hunt for bugs to create dye" instead of revenge (PCGamesN); but
  faction lines contain dungeons, puzzles and bosses.
  https://www.thegamer.com/crimson-desert-review-round-up/ ,
  https://gameinformer.com/review/crimson-desert/open-world-overload ,
  https://www.pcgamesn.com/crimson-desert/review
* Some side content locks after Chapter 11 `[single-source]`. https://pyweldb.com/guides/story-chapters/

### 5.2 Dialogue, choices, reputation, law

* No player-authored dialogue; a limited set of choices sway whether factions become allies or
  enemies; endings do not branch.
  https://steamcommunity.com/app/3321460/discussions/0/761806896608981679/
* **Faction reputation**: five tiers (Hostile → War … Friendly); killing faction NPCs drops standing;
  flagged quest choices permanently shift standing; ~110 faction knowledge entries / 76 lore
  entries (19 core, 21 hostile, 4 religious, 32 other) `[uncertain counts]`; "hostile does not mean
  permanently hostile" (disguises, cleared strongholds, story state).
  https://thegameswiki.com/crimson-desert/wiki/faction-reputation ,
  https://consolepulse.com/multiplatform/crimson-desert/guides/crimson-desert-factions-guide
* **Contribution** (per region, MMO-style "Hero Contribution"): helpful acts fill a bar → points
  spent at regional Contribution Shops (4–70 pts; armour, horse gear, banners; full refund on
  sell-back); also gates camp upgrades and merchants.
  https://www.dexerto.com/wikis/crimson-desert/contribution-explained/ ,
  https://thegameswiki.com/crimson-desert/wiki/contribution-system ,
  https://www.vice.com/en/article/crimson-desert-review-the-most-ambitious-open-world-game-since-red-dead-redemption-2/
* **Crime/law**: wear a mask to steal/pickpocket; each crime = Contribution penalty + red detection
  zone + timer; witnessed crimes create a Bounty Notice, guards patrol, region shaded; escalates to
  town-wide hostility; clear by jail time, Contribution, or a church "Writ of Absolution"; black
  market buys stolen goods, poached livestock, stolen wagons.
  https://steamcommunity.com/sharedfiles/filedetails/?id=3689585522 ,
  https://game8.co/games/Crimson-Desert/archives/586340 ,
  https://thegameswiki.com/crimson-desert/wiki/black-market

### 5.3 Economy and loot

* Currencies: copper (100 = 1 silver), silver, gold bars (bank: 500 silver; merchants only 190);
  separate **camp funds**; banking/investing. https://thegameswiki.com/crimson-desert/wiki/gold-and-currency-guide
* Trading unlocks in stages from Chapter 3 via the Greymane line: regional vendor inventories,
  trade goods, caravans/wagons, faction-gated merchants; black market parallel economy.
  https://thegameswiki.com/crimson-desert/wiki/trading ,
  https://steamcommunity.com/sharedfiles/filedetails/?id=3707066371
* Loot: quests, boss drops, vendors, chests; no explicit rarity grades found — power comes from
  refinement tier and Abyss Gear sockets rather than colour rarity `[uncertain]`.
  https://beebom.com/crimson-desert-weapons/ , https://www.pywel.app/en/guides/abyss-system/

### 5.4 Liberation, sieges, dynamic world

* **Base liberation**: red-marked bandit camps, blockades, fortified quarries, military outposts;
  clear all enemies + commander; rewards artifact-gauge fill, Contribution, map access; residents
  return, vendors open, structures rebuild. Reviewers found them long/repetitive with respawns.
  https://vulkk.com/2026/03/31/how-to-liberate-and-clear-locations-in-crimson-desert/ ,
  https://thegameswiki.com/crimson-desert/wiki/base-liberation-guide ,
  https://www.destructoid.com/reviews/crimson-desert-review/
* **Siege battles**: dozens of allied/enemy soldiers, siege weapons, destructible walls/gates,
  multi-phase objectives (Calphade Rebellion, Chapter 6). https://thegameswiki.com/crimson-desert/wiki/siege-battles
* Dynamic events: night-only spawns, storm-conditional mounts, weather-driven stealth, re-blockade
  (planned), ambient NPC routines; no evidence of a Radiant-style event generator `[uncertain]`.
  https://steamcommunity.com/sharedfiles/filedetails/?id=3689066736 ,
  https://crimsondesertwiki.net/articles/mount-guide-all-29-mounts-combat-capabilities-and-how-to-get-them

### 5.5 Mini-games and activities

* Card games Duo (2 cards) and Five-Card (poker-like pots); arm wrestling and hand wrestling (mash
  gauge); Rock-Paper-Scissors (best of 3); archery and gun shot contests (first to 10 hits); boxing
  and spear duels; wrestling matches (grapples only); horse/mount racing; fishing; hunting; gambling
  dens; puzzle solving. Rewards: coins, faction standing, items; cooldowns on replay.
  https://crimsondesertgame.wiki.fextralife.com/Minigames ,
  https://deltiasgaming.com/all-fun-games-and-activities-in-crimson-desert-and-their-locations/ ,
  https://www.gamespot.com/gallery/crimson-desert-duo-five-card-minigame/2900-7576/
* **Challenges**: 350+ optional objectives across exploration, combat, weapon mastery, life skills,
  mini-games. https://thegameswiki.com/crimson-desert/wiki/all-challenges

### 5.6 Puzzles, dungeons, collectibles

* 37 Ancient Ruins (environmental puzzle → Cresset + Artifact); 40 Abyss Restoration sky-island
  puzzles behind 8 Spires and gates/skybridges (Chaos Forest, Secret Garden, Root's End, Precipice of
  Truth); Sanctums (Witches questline; Fusion Reactor Cores + Kuku Pot puzzles, Antumbra corruption);
  100+ caves with ores/murals/artifacts. Puzzles praised as "mentally stimulating" (Kotaku round-up)
  and damned as "nigh impossible without guides" (Destructoid).
  https://thegameswiki.com/crimson-desert/wiki/ancient-ruins-and-their-puzzles ,
  https://vulkk.com/2026/06/14/abyss-challenges-list-in-crimson-desert-conqueror-of-the-abyss-achievement/ ,
  https://www.pywel.app/en/guides/abyss-system/ ,
  https://kotaku.com/crimson-desert-metacritic-scores-review-roundup-2000680110
* **Memory Fragments** (past-event visions via the Visione helmet from Chapter 2), constellations,
  treasure chests, Knowledge entries, 8 Bells, 141 Sealed Artifacts.
  https://thegameswiki.com/crimson-desert/wiki/memory-fragments ,
  https://consolepulse.com/multiplatform/crimson-desert/guides/crimson-desert-all-collectibles-complete-checklist

### 5.7 Stealth

* Proximity + line-of-sight model, no light/shadow meter; enemy vision cones on the minimap change
  colour when suspicious; crouch vignette shows reduced/hidden state; used for hunting, infiltration,
  crime, first strikes; fog helps. https://thegameswiki.com/crimson-desert/wiki/stealth

### 5.8 Death, save, time

* Death: no silver/item/Artifact loss; consumed heals are the cost. Boss death menu: **Instant
  Revive** (Palmar Pill, 30 % HP), **Retry** (refunds consumables used), **Give Up** (last checkpoint
  before the arena). https://thegameswiki.com/crimson-desert/wiki/death-penalty
* Save: 3 rotating autosave slots (≈ every 10 min and on combat enter/exit, objectives,
  liberations, new areas, key items) + 9 manual slots; "Save Anytime"; loading a manual save made in
  danger places you somewhere safe; Steam Cloud; Mac cross-save added 2.02.
  https://thegameswiki.com/crimson-desert/wiki/save-system ,
  https://game8.co/games/Crimson-Desert/archives/587674 ,
  https://www.gamewatcher.com/crimson-desert/guides/patch-notes-hub-and-roadmap-of-updates
* **New Game "carry-over"** (2.0): after all main quests, start a new game keeping Abyss Links and
  personal silver (not gear/camp funds). https://allthings.how/crimson-desert-2-0-update-how-the-new-game-plus-carry-over-works/

---

## 6. Narrative

* Premise: Kliff, a Greymane mercenary from Pailune, survives the Black Bears' ambush (leader
  **Myurdin**; comrade **Jian** killed), regroups survivors in Hernand, rebuilds the Greymanes at
  Howling Hill, and is drawn from a revenge plot into a continental conflict ending in the Abyss
  against **Umbra**. https://en.wikipedia.org/wiki/Crimson_Desert ,
  https://crimsondesert.fandom.com/wiki/Black_Bears , https://gamerant.com/crimson-desert-all-boss-fights-locations-quests/
* Official taglines: "Reclaim Power. Reach Beyond." / "The Beginning of a New Adventure" / "a mission
  unlike any he has ever known". https://crimsondesert.pearlabyss.com/
* Factions: **Greymanes** (core; "protect Pailune and accept dangerous work across Pywel");
  **Black Bears** (hostile); Church of Solumen and the politically separate Hernandian Parish;
  **Antumbra Order** (dark religion corrupting Sanctums); Witches (Elowen, Bari, Lyselia…) who grant
  Abyss Gear; noble houses Celeste, Serkis, Lanford, Roberts, Felix, Wells, Azerian, Byron, Marshell,
  Thorel, Elemore, Grace; Goldleaf Merchant Guild / Hornsplitter's Guards; Redwind Merchant Guild;
  Kharonso Troll Alliance; Pailune Militia, Skoghorn, Longleaf/Odeck tribes, Stjar Clan; Delesyia's
  Marni, Society of Progress, National Institute, Cogknights, Aerial Force; Lords of Unclaimed Lands
  (desert). https://consolepulse.com/multiplatform/crimson-desert/guides/crimson-desert-factions-guide ,
  https://gamerant.com/crimson-desert-all-boss-fights-locations-quests/
* Tone: shifts "between George R.R. Martin-style fantasy and sci-fi shenanigans" with villain
  monologues (GameSpot); "delightful absurdity"; mercenary survival framing from the 2020 reveal.
  https://www.gamespot.com/reviews/crimson-desert-review-highest-fantasy/1900-6418474/ ,
  https://en.prnasia.com/releases/apac/pearl-abyss-unveils-crimson-desert-trailer-commentary-featuring-executive-producer-and-founder-daeil-kim-303421.shtml
* Reception: story "fatally undercooked" (Eurogamer), "simply a mess… nonsensical" (Game Informer),
  "I have no idea who [Kliff] is" (PCGamesN); the Greymane reunions are "the game's sole emotional
  core". Devs concede the world-first pipeline hurt cohesion: "some players felt the story
  progression wasn't as intense or cohesive as they hoped." The free 2.0 update added cutscenes,
  dialogue, knowledge entries and items to patch narrative flow.
  https://www.pushsquare.com/features/round-up-crimson-desert-reviews-are-a-major-disappointment ,
  https://gameinformer.com/review/crimson-desert/open-world-overload ,
  https://www.invenglobal.com/articles/24095/pearl-abyss-unveils-crimson-desert-development-process ,
  https://noisypixel.net/crimson-desert-enhanced-version-2-0-update/
* Presentation: real-time in-engine cutscenes affected by weather/time; strong voice acting
  salvages a "dull" script (Push Square); 70+ voice actors; English leads Alec Newman (Kliff),
  Rebecca Hanssen (Damiane), Stewart Scudamore (Oongka), Alastair Parker (Myurdin); original VO in
  English/Korean/Chinese plus five languages added in 2.0.
  https://www.pushsquare.com/reviews/ps5/crimson-desert ,
  https://gfuel.com/blogs/news/crimson-desert-all-main-voice-actors ,
  https://store.epicgames.com/en-US/news/crimson-desert-voice-actor-interview
* DLC **"Charting the Unknown"** (announced State of Play 3 Sep 2026; $24.99; 15 Oct → 29 Oct 2026):
  Kliff, Damiane and Oongka sail beyond Pywel; naval combat, underwater exploration, more mechs,
  expanded housing. https://en.wikipedia.org/wiki/Crimson_Desert , https://store.steampowered.com/app/3321460/Crimson_Desert/

---

## 7. Presentation

* Art direction: "playable nature" over photoreal nature — four criteria (guide the gaze into the
  distance, discoverable landmarks, broad sightlines, space that supports traversal and combat);
  zone-function vegetation density; quiet-before-loud contrast ("if every part is interesting,
  paradoxically nothing feels special"); landmarks designed twice (1 km silhouette, 10 m approach);
  skyline tuned across dawn/noon/storm.
  https://www.invenglobal.com/articles/25076/if-you-can-see-it-you-should-be-able-to-go-there-the-secrets-behind-the-art-of-pearl-abyss-crimson-desert
* World judged "gigantic scale and gleaming prettiness" but lacking "distinctiveness" (Eurogamer);
  "one of the best open worlds ever created" (Destructoid).
  https://www.pushsquare.com/features/round-up-crimson-desert-reviews-are-a-major-disappointment ,
  https://www.destructoid.com/reviews/crimson-desert-review/
* HUD: minimap with cardinal directions (1.04), temperature gauge, day/time, HP/stamina/spirit
  bars, element dial, consumable wheel; patch 1.13 lets players hide minimap/HP/status icons; HUD
  can shrink but not enlarge (criticised); inventory category tabs added 1.04; menus "buried inside
  menus" (Push Square); mods target inventory UI.
  https://vulkk.com/2026/07/04/how-to-hide-the-minimap-and-hp-bar-in-crimson-desert/ ,
  https://wccftech.com/crimson-desert-patch-1-04-difficulty-settings-controls-visuals/ ,
  https://gamingbolt.com/crimson-desert-is-better-than-before-yet-these-15-issues-still-hurt-it ,
  https://www.pushsquare.com/reviews/ps5/crimson-desert
* Audio: Executive Audio Director **Hwiman Ryu** (experimental electronic/percussion + hybrid
  orchestra), Music Director **Inro Joo**, composers Hyoung Woo Roh, Jiyoon Kim, Dongjune Oh; OST Vol.
  1 = 75 tracks, free on Steam/Epic; reviewers praise foley.
  https://store.steampowered.com/app/4572870/Crimson_Desert_Original_Soundtrack_Volume_1/ ,
  https://vgmdb.net/album/159649 , https://www.pcgamesn.com/crimson-desert/review
* Performance: PC widely praised (native 4K 65–70 fps on high-end — Destructoid); base PS5 criticised
  for tearing/fluctuation and PS5 Pro for PSSR artefacting and foliage pop-in (Push Square); five
  crashes in 100 h (Game Informer).
  https://www.destructoid.com/reviews/crimson-desert-review/ ,
  https://www.pushsquare.com/reviews/ps5/crimson-desert ,
  https://gameinformer.com/review/crimson-desert/open-world-overload
* Accessibility: colourblind mode, chromatic-aberration toggle, photosensitive mode (1.04); control
  presets; difficulty modes; subtitles; no UI scaling up. https://wccftech.com/crimson-desert-patch-1-04-difficulty-settings-controls-visuals/
* Photo mode: present; crash hotfix 1.12.02; camera tilt (2.01), P hotkey, slow-move modifier and
  inverted-axis parity (2.03). https://vulkk.com/2026/09/04/crimson-desert-update-2-01-improves-photo-mode-camp-funds-visual-noise-bugfixes/ ,
  https://vulkk.com/2026/09/18/crimson-desert-update-2-03-changes-overview/
* Mods: hundreds on Nexus (UI, camera, inventory, save editors); no official tools ("would require
  disclosing a significant portion of the engine"), not blocked.
  https://www.dexerto.com/wikis/crimson-desert/does-crimson-desert-have-mod-support ,
  https://insider-gaming.com/crimson-desert-developer-addresses-official-mod-support/

---

## 8. Scale and scope table

| Metric | Value | Confidence / source |
|---|---|---|
| Surface map | ~90 km² (9.5 × 9.5 km); ~100 km² with underground/sky | PowerPyx, Beebom |
| Regions | 5 surface + The Abyss dimension | Beebom, PowerPyx |
| Named landmarks | 780+ (Hernand 240 … Pailune 111) | mapmaster.io `[single-source]` |
| Caves | 100+ | PowerPyx |
| Ancient Ruins / Cressets | 37 | thegameswiki, pywel.app |
| Sky-island (Abyss Restoration) puzzles | 40 behind 8 Spires | VULKK |
| Map bells | 8 | PowerPyx |
| Sealed Abyss Artifacts | 141 | pywel.app, Steam guide |
| Main story | Prologue + 12 chapters + Epilogue; ~168 main quests | pyweldb `[single-source]` |
| Side "adventures" | 430 (Pearl Abyss figure); faction quests 186–280+ | Beebom, crimsondb |
| Factions | 76 lore entries / ~110 knowledge entries | consolepulse, thegameswiki `[uncertain]` |
| Knowledge entries | 2,921 | thegameswiki |
| Challenges | 350+ | thegameswiki |
| Bosses | 75–80 incl. 3 super-bosses | Game Rant, crimsondb |
| Weapons | 535+ items across 14 categories | crimsondb, Fextralife |
| Skills | Kliff 80+, Damiane ~70, Oongka ~70 (+8 Kliff in 2.0) | Steam guide |
| Mounts | 29 in 8 categories | crimsondesertwiki.net `[single-source]` |
| Cooking recipes | 40+ with 4 quality tiers | KeenGamer |
| Inventory | 50 → 240 slots; chests 230 / 1,000 | PowerPyx, WCCFTech |
| Refinement | 10 levels, ~+8 %/level | KeenGamer |
| Playable characters | 3 | GamesRadar |
| Voice cast | 70+ actors; 8 VO languages after 2.0 | thegameswiki, Pearl Abyss |
| OST | 75 tracks (Vol. 1) | VGMdb |
| Hours | dev ~50 main; players 60–80 main, 80–100+ with sides, 100–400 h completionist | icy-veins, Game8 `[uncertain]` |
| Engine world units | 20,000+ units; 500k vegetation/scene; 100+ tree species; 256 m sectors; 5 LODs | Inven Global |
| Team / time | ~200 (≤300) people, ~7 years | Inven Global |
| Install | 150 GB SSD listed; ~121 GB on disk | Steam, space4games |
| Sales | 2 M/24 h; 6 M/3 months | Wikipedia, MMORPG.com |
| Reviews | MC 77–78; OC 79–81, 75 % recommend | Metacritic/OpenCritic |

---

## 9. Design pillars (Pearl Abyss's words) and what reviewers consider defining

### Pearl Abyss's stated pillars

1. **Exploration is the gameplay.** "We wanted the open world itself to be the primary source of
   gameplay." / "Exploration itself had to feel like real gameplay, with quests adding context and
   narrative on top of that." — Kim Hyun-kyum, Design Office lead.
   https://www.invenglobal.com/articles/24095/pearl-abyss-unveils-crimson-desert-development-process ,
   https://otakukart.com/crimson-desert-devs-say-their-200-person-team-built-the-open-world-first-then-added-quests-exploration-itself-had-to-feel-like-real-gameplay/
2. **"If you can see it, you should be able to go there."** / "Exploration takes priority over
   beauty." — Ahn Geun-tae, Art Level head.
   https://www.invenglobal.com/articles/25076/if-you-can-see-it-you-should-be-able-to-go-there-the-secrets-behind-the-art-of-pearl-abyss-crimson-desert
3. **A "playable world", not a replicated one** — density and layout adjusted for movement and
   readability rather than realism. https://www.invenglobal.com/articles/25231/pearl-abyss-shares-crimson-desert-development-philosophy-at-gamescom-dev
4. **One continuous shared state**: "time, weather, lighting, water, and vegetation must be
   maintained as parts of a single shared state." — Kim Jin-hwan, Game Engine Graphics head. (same URL as 2)
5. **Expressive, not punishing, combat**: "combo-based controls, fluid transitions, and player-driven
   skill expression"; "not designed as a Souls-like game."
   https://www.gamereactor.eu/we-talk-difficulty-scale-and-inspirations-in-crimson-desert-with-pearl-abyss-1674833/
6. **A living world with purpose**: "all villagers going about their daily lives and every NPC, event,
   and side quest in the game were created for a purpose."
   https://www.gamespot.com/articles/crimson-desert-players-love-watching-npcs-work-in-real-time/1100-6539161/
7. **Physics and interaction as foundation**: "objects break depending on the force applied, and
   enemies react naturally to the environment." https://www.techporn.ph/pearl-abyss-unveils-blackspace-engine-at-gdc-2025/
8. **Do what the studio never tried**: "We wanted to focus on things we never tried before" —
   adventure and exploration themes (Daeil Kim). https://en.prnasia.com/releases/apac/pearl-abyss-unveils-crimson-desert-trailer-commentary-featuring-executive-producer-and-founder-daeil-kim-303421.shtml
9. **Finite single-player journey**, "a defined beginning and end", not live service. (Gamereactor URL above)
10. **Production: fast iteration, maintainability, parallel workflows; data-driven tools so designers
    never wait on engineers.** https://www.invenglobal.com/articles/24095/pearl-abyss-unveils-crimson-desert-development-process

### What reviewers consider its defining features

* Scale + freedom: "the most ambitious open world I've ever experienced"; BotW-style landmark pull
  with far denser side content (Vice, Push Square, Game Informer).
* Combat: wrestling-plus-swordplay hybrid ("Devil May Cry meets WWE"), parry/counter economy,
  physics throws; "killer combat".
* Systems maximalism: "the 'Yes, and' of videogames… stuffed with just about every mechanic that has
  ever existed" (PC Gamer); "a game for the sickos".
* Technical achievement: seamless world, weather, physics, PC performance.
  https://www.vice.com/en/article/crimson-desert-review-the-most-ambitious-open-world-game-since-red-dead-redemption-2/ ,
  https://www.thegamer.com/crimson-desert-review-round-up/ ,
  https://www.pushsquare.com/reviews/ps5/crimson-desert

### What reviewers consider its weaknesses

* Story and characters ("fatally undercooked", "nonsensical"); MMO-style busywork quests; obtuse
  onboarding ("dense and cryptic"); slot-limited inventory and nested menus; lock-on/camera and
  "wonky" controls; boss difficulty spikes and gimmick fights; liberation grind; bloat ("not all of
  that is fun"); console performance; some puzzles need guides; one patch regressed performance.
  https://www.pushsquare.com/features/round-up-crimson-desert-reviews-are-a-major-disappointment ,
  https://kotaku.com/crimson-desert-metacritic-scores-review-roundup-2000680110 ,
  https://gameinformer.com/review/crimson-desert/open-world-overload ,
  https://www.pcgamer.com/games/action/crimson-desert-launches-to-239-000-players-on-steam-but-mixed-reviews-and-its-mostly-because-of-how-dense-and-cryptic-the-whole-thing-is/
* Not located: a Rock Paper Shotgun review and a Polygon review (searches returned none; OpenCritic
  listing did not show them). VG247 reviewed without a score.
  https://opencritic.com/game/19373/crimson-desert/reviews

---

## 10. Comparable games and what each contributed (for scoping)

| Comparable | What Crimson Desert borrows / is compared on | Source |
|---|---|---|
| Zelda: Breath of the Wild / Tears of the Kingdom | Go-anywhere landmark pull, stamina-gated climbing of any surface, gliding, no hand-holding; sky islands and physics/telekinesis (Axiom Force ≈ Ultrahand) | Push Square review; PowerPyx (TotK sky islands); Vice |
| Elden Ring / Sekiro | Parry-timing boss fights and difficulty spikes that reviewers measured against Souls; devs explicitly disclaim "Souls-like" | Vice; Gamereactor; ScreenRant ("Sekiro… rolled into one") |
| Dragon's Dogma 2 | Climbing onto large monsters, mounted/colossal boss design; "plays a lot like Dragon's Dogma 2 without the Pawn system" | MMORPG.com Gamescom 2024; Tom's Guide |
| The Witcher 3 | Abundance of side questlines; region-by-region story progression; investigation quests | Vice; pyweldb |
| Red Dead Redemption 2 | Lived-in NPC schedules, foley, horse trust/bonding, crime/bounty/wanted system, camp management, ground-level care in towns | Vice ("most ambitious since RDR2"); Push Square; thegameswiki horse guide |
| Ghost of Tsushima | Not cited by the sources reviewed; nearest overlap is the stance/parry + minimal-HUD toggle (1.13) `[analyst inference]` | — |
| Shadow of the Colossus | Climb-the-boss weak-point fights (White Horn, Queen Stoneback Crab) | ScreenRant; GameSpot |
| Devil May Cry / fighting games / WWE | Combo expression, "more like Street Fighter", suplex/chokeslam grapples | Vice; ScreenRant; Windows Central |
| GTA V | Instant switching between three playable characters | mein-mmo |
| Black Desert Online (own lineage) | Life skills, contribution points, trading, horse taming, node-style economy; "MMO DNA" and archaic UI | PC Gamer via TheGamer round-up; Gamereactor |

URLs: https://www.pushsquare.com/reviews/ps5/crimson-desert ,
https://www.vice.com/en/article/crimson-desert-review-the-most-ambitious-open-world-game-since-red-dead-redemption-2/ ,
https://screenrant.com/crimson-desert-gamescom-2024-hands-on-action-rpg/ ,
https://www.mmorpg.com/previews/gamescom-2024-preview-we-try-out-crimson-desert-and-fall-on-our-swords-2000132649 ,
https://www.tomsguide.com/gaming/crimson-desert-review ,
https://mein-mmo.de/en/crimson-desert-the-3-playable-characters-and-their-characteristics-which-one-suits-you,1551179/ ,
https://www.thegamer.com/crimson-desert-review-round-up/

---

## SYSTEMS INVENTORY

Flat checklist of distinct systems/features found in sources above. One line each; grouped by area;
`*` = post-launch addition; `?` = single-source or uncertain.

### A. World, streaming, environment
1. Seamless continent streaming by position/FOV (BlackSpace), 256 m sectors, proxy LOD hierarchy
2. Three vertical layers: surface, underground caves/sanctums, floating sky islands
3. Five biome regions with fixed climates + an alternate-dimension region (Abyss)
4. Runtime weather model (biome/temperature/elevation/wind/time) — rain, snow, fog, sandstorm, blizzard
5. Day/night cycle with in-game calendar (day number, weekday, clock) and time-skip via sleep/bonfire
6. Ambient temperature with body-temperature gauge and stamina penalties; gear/food/heat-source resistance
7. Wind simulation affecting vegetation, cloth, hair, glide distance
8. Volumetric fog/clouds and ray-traced GI shared with gameplay (fog lowers enemy detection)
9. FFT ocean + shallow-water simulation; partial-wetting of characters/animals
10. Systemic destruction (walls, towers, carts, trees) scaled by applied force
11. Fire propagation across flammable materials and burning enemies
12. Interactive "gimmick" objects with data-defined states (levers, barrels, totems, thorns)
13. Phase-switched world states (persistent base level + mutable gameplay level) for rebuilt towns
14. Fog-of-war map cleared by 8 bells or exploration; map filters, search, quest-colour markers*
15. 780+ named landmarks with silhouette-first design (1 km / 10 m passes)
16. Night-only / weather-conditional enemy and mount spawns
17. NPC daily schedules (wake, work, sleep) and shop hours tied to the clock
18. NPC rain-sheltering and ambient work animations (crowd life)
19. Real-time cutscenes inheriting current time/weather

### B. Traversal and mounts
20. Stamina-gated free climbing of almost any surface
21. Double jump + aerial Force Palm triple-boost
22. Crow-Wing glider (always descending; stamina for speed) / Oongka rocket pack
23. Axiom Force grapple: anchor swing, object/enemy telekinesis, Aerial Maneuver/Swing upgrades
24. Swimming with drowning on empty stamina (no diving until DLC)
25. Mid-action consumable wheel (eat while climbing/swimming/gliding)
26. Horse taming minigame; registration/summon; trust levels 1–5 unlocking dash/combat; 4 gear slots
27. Exotic mounts: bears, wyverns (timed flight + cooldown), dragons, raptors/lizards/dinosaurs?
28. Crafted mechanical mounts/war mechs from blueprints and boss keys
29. Vehicles: wagons, skiffs/boats, hot-air balloons; caravan hijacking
30. Fast travel via activated Abyss Nexus plates and puzzle-gated Cressets; disallowed when mounted/in combat/swimming
31. Blinding Flash / Guiding Light detection pulses for nearby Abyss points
32. Mounted combat and mounted ranged lock-on*

### C. Player progression
33. Three core stats (Health, Stamina, Spirit) raised with Abyss Artifacts (no XP levels)
34. Abyss Artifact economy: regular (1 SP), Sealed (141, challenge-locked), Faded (full respec)
35. Uncapped kill gauge that pays out Artifacts
36. Per-character skill trees (~80/70/70 skills) with ranks
37. Watch-and-Learn: unlock enemy moves by observing/being hit
38. Story-gated skills (+8 Kliff skills in 2.0*)
39. Abyss Links cross-character skill resource and per-character resets*
40. Knowledge codex (2,921 entries) that also gates some upgrades
41. 350+ challenge objectives (exploration, combat, weapon mastery, life skills, mini-games)
42. Three playable characters with instant switching and distinct kits
43. New-game carry-over of Abyss Links and personal silver after the campaign*

### D. Equipment, inventory, crafting
44. 14 weapon categories incl. firearms, hand cannon, unarmed, mount weapons; 535+ weapon items
45. Quick-Swap secondary weapon slot (Kliff)
46. Shields with parry/block/counter off-hand behaviour
47. Armour sets, cloaks with elemental resistance, faction/contribution cosmetic sets
48. Slot-based inventory (50→240) grown by purchasable/quest bags
49. Camp Supply Chest (230) and specialised 1,000-slot chests, coolers, wardrobes*
50. Refinement +1…+10 (~8 %/level), material tiers, Artifact cost at 5+
51. Abyss Gear/Core sockets (Swift/Destruction/Fortification + unique boss cores), socket count by tier
52. Dye station and outfit cosmetics; hide-back-weapons toggle*
53. Cooking at any fire: free-form ingredient combos, recipe memorisation, 4 quality tiers, 40+ recipes
54. Alchemy (attack-speed, spirit potions), Palmar Pill revive consumable
55. Gathering life skills with proficiency: mining, herbalism, logging, fishing, hunting
56. Workstations (wood/stone), looms (textiles), blacksmith/forge crafting of top-tier gear

### E. Camp, housing, economy
57. Greymane Camp base-building with recruit-gated facilities and upgrade tiers
58. Recruit roster with morale affecting companion performance
59. Dispatch missions (solo and Duo) for loot/Contribution
60. Farming (watering/fertiliser) and ranching (capacity, breeding, feed)
61. Player housing with furniture placement, multiple layouts, bulk retrieval*
62. Camp funds separate from personal copper/silver/gold-bar currencies; bank exchange and investing
63. Staged trading system (trade goods, regional vendors, caravans) unlocked via Greymane quests
64. Black market for stolen goods, poached livestock, stolen wagons
65. Regional Contribution bars/points and Contribution Shops (refundable purchases)
66. Faction-gated merchant access and region-locked reputation

### F. Combat
67. Light/heavy attack strings per weapon category with stamina-free lights
68. Timed parry with 2.5× counter window, Stamina/Spirit refund, unblockable red attacks
69. Perfect dodge/backstep (Keen Senses) and Counter (pre-emptive strike) skills
70. Grapple suite: pin, wall slam, suplex AoE, lariat, throw, back-hang, clothesline
71. Stagger/super-armour model with multi-bar bosses; boss counter/escape frequency tuning*
72. Execution/finisher animations
73. Elemental imbue dial (fire/lightning/ice) with burn/stagger/slow status interactions and terrain spread
74. Spirit-fuelled active skills (Force Palm, Blinding Flash, Frost Mantle, Lightning Pulse, Storm Howl…)
75. Ranged kit: bow draw, pistol/rifle/hand cannon, special siege arrows
76. Environmental combat: throw enemies into props/cliffs, explosive barrels, collapsing structures
77. Enemy-contact environmental damage (tuned down 1.04)
78. Soft/hard lock-on with Manual/Semi-auto/Auto camera rotation modes*
79. Control presets (Classic/new), dodge input style, vault rework*
80. Difficulty modes Easy/Normal/Hard changing windows, i-frames, food timing, boss patterns*
81. Enemy archetypes: bandits, elites that parry, warlords, pack hunters, heavy beasts, colossal weak-point creatures, goblin formations, undead that resurrect, sorcerers, automatons/mechs
82. 75–80 bosses incl. climb-and-destroy colossi, illusion-clone duelists, mech dragons, 3 super-bosses
83. Boss death menu: instant revive / retry with refund / give up to checkpoint
84. Boss rematch mode (announced)? 
85. Siege battles with allied armies, siege weapons, destructible fortifications, multi-phase objectives
86. Base liberation (camps, blockades, quarries, outposts) with world-state rebuild rewards
87. Re-blockade counter-attacks on liberated areas (announced)?

### G. Quests, narrative, factions
88. Prologue + 12 chapters + epilogue main campaign
89. Faction questlines per house/guild/tribe/church (186–280 quests)
90. Commission/notice boards per region
91. Bounty hunting (outlaws) and the Wanted/bounty state applied to the player
92. Investigation/contradiction dialogue puzzles
93. Research quests (institute upgrades, e.g. stamina cap)
94. Daily-life and constellation activities
95. Organic "rumour" quest discovery (overheard/found leads)
96. Limited binary dialogue choices shifting faction alliance (no ending branches)
97. Five-tier faction reputation with crime-driven decay and permanent story shifts
98. Crime/law loop: masks, detection zones, timers, witnesses, guard patrols, jail, Writ of Absolution
99. Disguise system (mask) for infiltrating hostile territory
100. Memory Fragments (Visione helmet) lore visions
101. Journal with reward previews; quest knowledge entries*
102. Post-launch narrative patching (new cutscenes/dialogue in 2.0)*

### H. Activities, puzzles, collectibles
103. Card games (Duo, Five-Card), Rock-Paper-Scissors, gambling dens
104. Arm/hand wrestling mash minigames
105. Archery and gun shooting contests, mount racing
106. Rule-limited duels (boxing, spear, wrestling)
107. Fishing and hunting loops feeding cooking/crafting
108. 37 Ancient Ruins environmental puzzles → Cresset + Artifact
109. 40 sky-island Abyss Restoration puzzles behind 8 Spires, gates and skybridges
110. Sanctum dungeons (Fusion Reactor Core / Kuku Pot puzzles, Antumbra corruption)
111. 100+ caves with ores, murals, hidden artifacts, treasure chests
112. Sealed Artifact shrine challenges (weapon-type kill counts etc.)
113. Pet taming by trust (pet/feed), auto-loot, Sigil of Valor combat pets, shoulder cats, bird sotdae*
114. Stealth: LOS/proximity detection, minimap vision cones, crouch vignette, first-strike bonuses

### I. UX, presentation, platform
115. Minimal-by-option HUD (hide minimap/HP/icons), element dial, consumable wheel, temperature gauge
116. Inventory category tabs and persistent sort*; map search/filters*
117. Accessibility: colourblind, chromatic-aberration toggle, photosensitive mode*
118. Photo mode with tilt, slow-move modifier, hotkey*
119. Save Anytime: 3 autosaves + 9 manual; safe-relocation on load; cloud and Mac cross-save*
120. 15 interface languages, 8 VO languages after 2.0, Arabic UI*
121. DLSS 4 / FSR Redstone / XeSS upscaling, frame generation, Reflex, HDR, RT quality tiers
122. Console graphics modes (Performance/Balanced/Quality) incl. PS5 Pro PSSR
123. Real-time in-engine cutscenes; 70+ voice actors; hybrid orchestral/electronic score (75-track OST)
124. Knowledge/lore codex UI with per-category completion counters
125. Achievements (e.g., "Conqueror of the Abyss") and Steam Cloud/Family Sharing
126. Unofficial mod ecosystem (Nexus) with no official tools
127. Paid expansion with naval combat, underwater exploration, mechs, housing (Charting the Unknown)*

**Inventory count: 127 items** (items 84 and 87 flagged as announced-only).
