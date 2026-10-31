import { getMeta } from "./api.js";
import { initScan, activateScan, deactivateScan } from "./tabs/scan.js";
import { initEnroll, activateEnroll, deactivateEnroll } from "./tabs/enroll.js";
import { initEvaluate, activateEvaluate } from "./tabs/evaluate.js";

const tabs = {
  scan: { activate: activateScan, deactivate: deactivateScan },
  enroll: { activate: activateEnroll, deactivate: deactivateEnroll },
  evaluate: { activate: activateEvaluate, deactivate: () => {} },
};
let activeTab = "scan";

function showTab(name) {
  if (name === activeTab) return;
  tabs[activeTab]?.deactivate?.();
  document.querySelectorAll(".tab").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll("nav button").forEach((el) => {
    el.classList.toggle("active", el.dataset.tab === name);
  });
  document.getElementById(`tab-${name}`).classList.add("active");
  activeTab = name;
  tabs[name]?.activate?.().catch((error) => {
    const status = document.getElementById(`${name}-status`);
    if (status) status.textContent = error.message;
  });
}

async function boot() {
  document.querySelectorAll("nav button").forEach((button) => {
    button.addEventListener("click", () => showTab(button.dataset.tab));
  });
  const meta = await getMeta();
  initScan(meta);
  initEnroll(meta);
  initEvaluate(meta);
  await activateScan();
}

boot().catch((error) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<p class="status error">Startup error: ${error.message}</p>`,
  );
});
