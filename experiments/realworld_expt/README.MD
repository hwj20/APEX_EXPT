
# 🤖 APEX: Real-World Reactive Planning on a Low-Cost Robot Arm

### 🧪 Experimental Setup

* **Robot:** HiWonder Mini Arm (5 DOF)
* **Compute:** Raspberry Pi 4B (Arm32, Python 3.7)
* **Camera:** Onboard RGB-D module (attached to end-effector)

---

### 🎯 Task Description

* **Objective:** Reactive collision avoidance
* **Scenario:** A human moves a red block toward a static blue block.
* **Goal:** The agent must detect the potential collision and move to prevent contact.
* **Prompt Format:** See `ApexAgent.py`
* **Baseline:** GPT-4o, queried every 1 second with no trigger or simulation.
* **Perception Method:** Classical CV techniques — color-based segmentation, bounding box tracking, and depth estimation.
* **Filtering:** 5-frame sliding window for position smoothing.
* **Metrics:** Response rate, collision rate, planning latency.
* **Simulation Model:** First-order Euler integration (analytical approximation).

---

### ⚙️ How to Run

1. Place all files under:

   ```bash
   /home/pi/ArmPi_mini/
   ```

2. Run the entry script with sudo:

   ```bash
   sudo python3 ApexExtension.py
   ```

---

### ⚠️ Notes

* This is a **very low-cost and low-resource** deployment. Due to hardware limitations (Python 3.7 on Arm32, no PyTorch, no modern MuJoCo), the graph-based trigger and full simulation were replaced with:

  * **Analytical collision prediction** based on geometry
  * **First-order Euler integration** to simulate motion

* We make this public despite expecting **very few attempts at reproduction**, as the hardware is limited and somewhat fragile. Still, we believe the structure and logic may offer value to the community.

---

### 📹 Demo Videos

All videos available at: **\[https://drive.google.com/drive/folders/1YMmYrXk8mii0v0LOx-QwHxOpq-Jq9rOO?usp=drive_link]**



