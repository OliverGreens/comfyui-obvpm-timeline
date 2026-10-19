// The Auto Chunk Plan's two pure decisions, LIFTED from web/h3_mctx_ui.js
// (not copied) and driven with the shapes the strip actually hands them:
// the gate ("why is that button greyed out?") and the landing rule ("where
// does this take go?"). Both are worth pinning down without a ComfyUI graph
// in the room:
//   node tests/test_auto_chunk_plan_gate.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const SRC = readFileSync(new URL("../web/h3_mctx_ui.js", import.meta.url), "utf8");
const lift = (name) => {
    const at = SRC.indexOf("function " + name + "(");
    assert.ok(at >= 0, name + " is gone from the source");
    // past the PARAMETER list first: this gate takes a destructured
    // object, and its braces are not the body's
    let i = SRC.indexOf("(", at), pd = 0;
    for (;; i += 1) {
        if (SRC[i] === "(") pd += 1;
        else if (SRC[i] === ")" && --pd === 0) break;
    }
    let depth = 0;
    i = SRC.indexOf("{", i);
    for (;; i += 1) {
        if (SRC[i] === "{") depth += 1;
        if (SRC[i] === "}" && --depth === 0) break;
    }
    return SRC.slice(at, i + 1);
};
const { tlAutoBlocked, autoLanding } = new Function(
    lift("tlAutoBlocked") + "\n" + lift("autoLanding")
    + "\nreturn { tlAutoBlocked, autoLanding };")();

// The plan shape chunkPlanCompute returns, trimmed to what the gate reads.
const planned = { next: { index: 3, role: "root", frames: 192 },
                  done: 2, total: 9 };

// ---- a plan with a chunk left is the one case that says go --------------
assert.equal(tlAutoBlocked({ upscale: false, plan: planned }), null);

// ---- upscale takes Runs away from generation ----------------------------
// The button is greyed while the upscale toggle is selected, even with a
// perfectly good plan: a Run in that mode refines the timeline, it does
// not generate chunks, so a loop would be refining the same piece forever.
let why = tlAutoBlocked({ upscale: true, plan: planned });
assert.ok(why, "upscale on must block the loop");
assert.match(why, /upscaling is on/);

// ---- chunk plan off: there is no schedule to run ------------------------
// lastPlan is null exactly when the chunk_plan widget is off, which is how
// the strip knows not to describe a plan at all.
why = tlAutoBlocked({ upscale: false, plan: null });
assert.ok(why, "no plan must block the loop");
assert.match(why, /chunk plan is off/);

// ---- the plan cannot be read yet, or at all -----------------------------
why = tlAutoBlocked({ upscale: false, plan: { pending: true } });
assert.match(why, /reading the soundtrack/);
why = tlAutoBlocked({ upscale: false,
                      plan: { error: "add a soundtrack on the audio row." } });
assert.match(why, /add a soundtrack/, "the plan's own reason is passed on");

// ---- complete: nothing left to queue ------------------------------------
why = tlAutoBlocked({ upscale: false,
                      plan: { next: null, done: 9, total: 9 } });
assert.match(why, /complete/);

// ---- the questions in order ---------------------------------------------
// Upscale outranks every plan question, and a plan that cannot be read
// outranks "is there anything left in it" -- the tooltip should name the
// first thing the user has to fix.
assert.match(tlAutoBlocked({ upscale: true, plan: { error: "x" } }),
             /upscaling is on/);
assert.match(tlAutoBlocked({ upscale: true, plan: null }), /upscaling is on/);
assert.match(tlAutoBlocked({ upscale: false, plan: { pending: true,
                                                     next: null } }),
             /reading the soundtrack/);
assert.match(tlAutoBlocked({ upscale: false, plan: { error: "boom",
                                                     next: null } }),
             /boom/);

// ---- where the take lands: the TAKE says, not the plan ------------------
// A bridge take reports relation "bridges" (or "extends"/"prepends" when
// only one of its two pins was recorded) and goes through the lineage
// placement, into the hole. A plain take is a root and goes past the last
// root with a hole behind it. The plan's role only decides for a take that
// arrived with no lineage -- and a take with lineage wins even when the
// plan is a step behind, which is what stranded a generated bridge outside
// the timeline once.
assert.equal(autoLanding("bridges", "bridge"), "linked");
assert.equal(autoLanding("bridges", "root"), "linked");
assert.equal(autoLanding("extends", "root"), "linked");
assert.equal(autoLanding("prepends", "root"), "linked");
assert.equal(autoLanding("", "bridge"), "linked");
assert.equal(autoLanding("", "root"), "root");
assert.equal(autoLanding("", null), "root");

console.log("auto chunk plan gate: all checks pass");
