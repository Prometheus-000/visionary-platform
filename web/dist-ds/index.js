import { jsx as l, jsxs as g } from "react/jsx-runtime";
import { useRef as x, useState as w, useCallback as f, useLayoutEffect as L, useEffect as k } from "react";
import { createPortal as E } from "react-dom";
function N({
  anchor: e,
  className: s,
  onClose: o,
  children: r
}) {
  const t = x(null), [i, c] = w(null), a = f(() => {
    const u = t.current;
    if (!u || !e) return;
    const n = e.getBoundingClientRect(), m = u.offsetWidth, d = u.offsetHeight, h = Math.max(8, Math.min(n.right - m, window.innerWidth - m - 8)), b = n.bottom + d + 10 > window.innerHeight ? n.top - d - 6 : n.bottom + 6, v = Math.max(8, Math.min(b, window.innerHeight - d - 8));
    c((p) => p && p.left === h && p.top === v ? p : { left: h, top: v });
  }, [e]);
  return L(a), k(() => {
    const u = (d) => {
      !t.current?.contains(d.target) && !e?.contains(d.target) && o();
    }, n = (d) => {
      t.current?.contains(d.target) || o();
    }, m = (d) => {
      d.key === "Escape" && o();
    };
    return document.addEventListener("mousedown", u, !0), document.addEventListener("scroll", n, !0), document.addEventListener("keydown", m), window.addEventListener("resize", o), () => {
      document.removeEventListener("mousedown", u, !0), document.removeEventListener("scroll", n, !0), document.removeEventListener("keydown", m), window.removeEventListener("resize", o);
    };
  }, [e, o]), E(
    /* @__PURE__ */ l(
      "div",
      {
        ref: t,
        className: s,
        style: { left: i?.left ?? 0, top: i?.top ?? 0, visibility: i ? "visible" : "hidden" },
        children: r
      }
    ),
    document.body
  );
}
function B() {
  const [e, s] = w(null), o = f(() => s(null), []);
  return {
    anchor: e,
    open: !!e,
    close: o,
    /** Pass as `onClick`. Toggles, so a second press on the button that opened it
     *  closes it — which is what anyone does when they meant to look and not to
     *  choose. */
    toggle: f((r) => {
      const t = r.currentTarget;
      s((i) => i === t ? null : t);
    }, []),
    /** Open against an element you already have. The gallery's card menu needs this: the
     *  anchor is the card's own ⋯ button, but *which item* the menu is about is state the
     *  card cannot hold, so the handler sets both and cannot go through `toggle`'s
     *  event. */
    at: f((r) => s(r), [])
  };
}
function S({
  anchor: e,
  items: s,
  onClose: o
}) {
  const r = s.some((t) => !t.sep && t.on);
  return /* @__PURE__ */ l(N, { anchor: e, className: `menu${r ? " checks" : ""}`, onClose: o, children: s.map(
    (t, i) => t.sep ? /* @__PURE__ */ l("hr", {}, `sep${i}`) : /* @__PURE__ */ g(
      "button",
      {
        type: "button",
        className: [
          t.danger ? "danger" : "",
          t.on ? "on" : "",
          t.drag ? "draggable" : ""
        ].filter(Boolean).join(" "),
        draggable: !!t.drag,
        onDragStart: t.drag && ((c) => {
          t.drag(c), setTimeout(o, 0);
        }),
        onClick: () => {
          o(), t.run();
        },
        children: [
          t.label,
          t.hint && /* @__PURE__ */ l("span", { className: "hint", children: t.hint })
        ]
      },
      t.label
    )
  ) });
}
function T({
  onClose: e,
  children: s,
  id: o
}) {
  return k(() => {
    const r = (t) => {
      t.key === "Escape" && e();
    };
    return document.addEventListener("keydown", r), () => document.removeEventListener("keydown", r);
  }, [e]), E(
    /* @__PURE__ */ l(
      "div",
      {
        className: "scrim",
        id: o,
        onClick: (r) => {
          r.target === r.currentTarget && e();
        },
        children: /* @__PURE__ */ l("div", { className: "sheet", children: s })
      }
    ),
    document.body
  );
}
const y = (e) => (String(e).split(".")[1] ?? "").length;
function M(e, s, o, { fine: r = 1, bigStep: t, base: i } = {}) {
  const c = o ? t ?? r * 8 : r, a = e !== "" ? parseFloat(e) : i;
  if (a === void 0 || !Number.isFinite(a)) return null;
  const u = Math.max(0, Number((a + s * c).toFixed(Math.max(y(c), y(a)))));
  return String(u);
}
function j({
  value: e,
  onValue: s,
  fine: o,
  bigStep: r,
  base: t,
  onEnter: i,
  onCommit: c,
  ...a
}) {
  const u = f(
    (n) => {
      if (n.key === "Enter" && !n.nativeEvent.isComposing) {
        n.preventDefault(), c?.(), n.currentTarget.blur(), i?.();
        return;
      }
      if (n.key !== "ArrowUp" && n.key !== "ArrowDown" || n.altKey || n.shiftKey) return;
      const m = M(
        e,
        n.key === "ArrowUp" ? 1 : -1,
        n.metaKey || n.ctrlKey,
        { fine: o, bigStep: r, base: t }
      );
      m !== null && (n.preventDefault(), s(m));
    },
    [e, s, o, r, t, i, c]
  );
  return /* @__PURE__ */ l(
    "input",
    {
      ...a,
      value: e,
      inputMode: a.inputMode ?? "decimal",
      onChange: (n) => s(n.target.value),
      onBlur: (n) => {
        c?.(), a.onBlur?.(n);
      },
      onKeyDown: u
    }
  );
}
function F({ err: e, style: s }) {
  if (!e) return null;
  const o = typeof e == "string" ? e : e.error, r = typeof e == "string" ? void 0 : e.detail;
  return !o && !r ? null : /* @__PURE__ */ g("div", { className: "err-box", style: s, children: [
    o,
    r && /* @__PURE__ */ g("details", { className: "err-detail", children: [
      /* @__PURE__ */ l("summary", { children: "What the server said" }),
      /* @__PURE__ */ l("pre", { children: r })
    ] })
  ] });
}
export {
  F as ErrorNote,
  S as Menu,
  j as NumInput,
  N as Popover,
  T as Sheet,
  M as step,
  B as usePopover
};
