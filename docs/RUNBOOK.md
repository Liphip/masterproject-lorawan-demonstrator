# Runbook — setup, demonstration, packing

## 1 · One-time laptop setup (Windows)

1. Install Python 3.12 (python.org, tick "py launcher"), Git and Git LFS.
2. `git clone <gitlab-url>` and `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1`
3. In an **admin** PowerShell, allow the gateway in (the setup script prints this command):
   `New-NetFirewallRule -DisplayName "LoRaWAN demonstrator 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Domain`
   Set the demo Wi-Fi network profile to **Private** in Windows.
4. Trained models in `data\models\`: run the notebooks (README, "Train the models") on any machine with the same
   scikit-learn version and commit `data/models/`, or train directly on the laptop with `.\.venv\Scripts\python.exe -m ml_core.training`.
5. Edit `config\demo.toml`: one `[[devices]]` block per sensor, with its DevEUI, type and slot.

## 2 · One-time gateway setup (Milesight UG63)

> Menu names follow the Milesight UG6x user guide. Check them against the installed firmware and fix this section after the first setup.

1. **Network:** connect the UG63 to the demo router (Ethernet or Wi-Fi client). In the router, give **the laptop** a fixed IP
   (DHCP reservation). The gateway URL depends on it.
2. **Packet forwarder:** region EU868. The embedded network server must be the forwarding destination.
3. **Network Server → General:** enable the embedded network server.
4. **Network Server → Applications → Add:**
   - Payload codec: select the vendor codec per device type if available (Milesight WS202 is usually built in;
     add the Dragino LSE01/LTC codecs as custom JS from the Dragino repository).
   - Data transmission: **HTTP**, uplink data URL `http://<laptop-ip>:8000/ingest/uplink`
     (add `?token=<value>` if `ingest_token` is set in the config).
5. **Network Server → Profiles / Devices:** add each sensor (OTAA: DevEUI, AppEUI/JoinEUI, AppKey).
   Make sure the sensors are **not also registered** on another network server that a nearby gateway serves (TTN, company LNS).
6. Power-cycle or rejoin the sensors. In the dashboard, an uplink appears in panel 1. A sensor shown as
   *not in config* has reached the laptop but has no entry in `config\demo.toml` yet.
7. Check one uplink per sensor type in `data\recordings\raw_YYYYMMDD.jsonl` against the vendor app or manual
   (units, decimal scaling). If a value field is named differently, add `fields = { <column> = "<key>" }` to the device.

## 3 · Demonstration

**Short version (presentation view, `http://localhost:8000/`, 2–3 min).** *Aufgezeichnete Wurst* → ▶: the phase
advances, the three models move with it, the "Metzger" tag shows the butcher's annotation. *Virtuelle Wurst*: drag the
day, then *trockener ↔ feuchter*. *Live-Sensoren*: set the room temperature, hold a probe. **Details →** leads to the
full dashboard below for questions.

**Full version (`/details`, about 10 min):**

| Step | Action | What to say |
|---|---|---|
| 0 · Before guests arrive | Power the router, then the gateway, then the laptop. Run `scripts\start_demo.ps1`. Pick the audience's language (DE/EN, top right). Set the air temperature of the room in the *Live sensors* tab (the case has no air sensor). Wait for a green "Gateway: last uplink …" chip. If the sensors send rarely, press the sensor button to force an uplink. | – |
| 1 · Data path | Point at the strip at the top and at panel 1 (sensor, values, RSSI). | LoRa radio → gateway (network server) → router (stands in for the Internet) → laptop. Nothing leaves the case. |
| 2 · AI comparison | Panel 2 → *Recorded run*. Pick a `series_1` sausage, 1 h/s, start. Show the three cards, the annotation card and the timeline lanes. | Replay because curing takes days. The models never saw this sausage. Look at where they disagree: at phase transitions. |
| 2b · Why? | At a phase transition press **❚❚ Pause**, then **🔍 How was this decided?** Walk through the 4 steps: uplinks → 29 numbers → which signals mattered → inside each model. **Close & resume replay** continues. | The same numbers, three principles: a weighted sum, 100 trees voting, a small neural network. Step 3 shows whether they rely on the same signals. |
| 2c · Virtual sausage | Tab *Virtual sausage (simulate)*. Move **Day of curing**, then try the presets *Dried out* / *Too warm*, or ask the audience what to change. Use **How was this decided?** on any setting. | There is no sausage in the case, so we take a real recorded day and ask "what if". The models react differently, and the out-of-range warning shows when we leave what they know. |
| 3 · Interpretation | Panel 3: balanced accuracy, latency (ms on CPU), model size. Then panel 4. | Small models are enough here: they run in milliseconds on a laptop with no GPU and no cloud. The scores come from one run, so they are optimistic. |
| 4 · Live, optional | Panel 2 → *Live sensors*. Hold the LTC-FS probe in your hand. Optionally put the LSE01 probe into a glass of tap water, then salt water (check the probe's IP rating first). | The live chain works end to end: watch conductivity jump within one uplink. The models flag this as outside the training range instead of inventing a curing phase. That is an honest system. |

Replaying the unlabeled `series_2_*` (an independent run, no annotation) shows the models on a run they have never seen.

## 4 · Shutdown and packing

1. Stop the replay. Press `Ctrl+C` in the console window.
2. Shut down the laptop, unplug the gateway, then the router.
3. Pack sensors in their foam slots and coil the cables. Check against the packing list (to be written during assembly, A5).
4. Optional: copy `data\recordings\` to the repository or a USB stick. Live recordings can be replayed later.

## 5 · Troubleshooting

| Symptom | Check |
|---|---|
| No uplinks in panel 1 | Laptop IP still the same? Firewall rule present and network profile *Private*? Open `http://<laptop-ip>:8000/healthz` from another device on the same Wi-Fi. Check the HTTP URL in the UG63 application. Are the sensors joined (UG63 device list)? |
| Device listed as "not in config" | DevEUI in `config\demo.toml` (case and separators do not matter). Restart the app after editing. |
| "no usable values … configure a payload codec" | Add the codec on the gateway, or add a `fields` override (see §2.7). |
| "Models: No models found" | Train the models (step 1.4). Run `git lfs pull` if the data files are 130-byte pointers. |
| "sklearn version mismatch" chip | Retrain on this laptop, or install the scikit-learn version shown in the chip tooltip. |
| Replay options say "LFS pointer" | `git lfs pull` |
| Test without gateway | `.\.venv\Scripts\python.exe scripts\send_test_uplink.py --rounds 12` |
