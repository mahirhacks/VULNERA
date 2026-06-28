import { createPortal } from "react-dom";
import { useLayoutEffect, useRef, useState } from "react";

export default function HistoryContextMenu({
  open,
  anchorRef,
  panelRef,
  children,
}) {
  const localPanelRef = useRef(null);
  const [coords, setCoords] = useState(null);

  useLayoutEffect(() => {
    if (!open || !anchorRef?.current) {
      setCoords(null);
      return;
    }

    const updatePosition = () => {
      const anchor = anchorRef.current;
      const panel = localPanelRef.current;
      if (!anchor) return;

      const rect = anchor.getBoundingClientRect();
      const menuHeight = panel?.offsetHeight ?? 48;
      const menuWidth = panel?.offsetWidth ?? 120;
      const margin = 8;
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;
      const openUp = spaceBelow < menuHeight + margin && spaceAbove >= spaceBelow;

      let top = openUp ? rect.top - menuHeight - 4 : rect.bottom + 4;
      top = Math.max(margin, Math.min(top, window.innerHeight - menuHeight - margin));

      let left = rect.right - menuWidth;
      left = Math.max(margin, Math.min(left, window.innerWidth - menuWidth - margin));

      setCoords({ top, left, openUp });
    };

    updatePosition();
    window.addEventListener("resize", updatePosition);
    return () => window.removeEventListener("resize", updatePosition);
  }, [open, anchorRef, children]);

  useLayoutEffect(() => {
    if (!open || !localPanelRef.current) return;
    const panel = localPanelRef.current;
    const observer = new ResizeObserver(() => {
      if (!anchorRef?.current) return;
      const rect = anchorRef.current.getBoundingClientRect();
      const menuHeight = panel.offsetHeight;
      const menuWidth = panel.offsetWidth;
      const margin = 8;
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;
      const openUp = spaceBelow < menuHeight + margin && spaceAbove >= spaceBelow;
      let top = openUp ? rect.top - menuHeight - 4 : rect.bottom + 4;
      top = Math.max(margin, Math.min(top, window.innerHeight - menuHeight - margin));
      let left = rect.right - menuWidth;
      left = Math.max(margin, Math.min(left, window.innerWidth - menuWidth - margin));
      setCoords({ top, left, openUp });
    });
    observer.observe(panel);
    return () => observer.disconnect();
  }, [open, anchorRef]);

  const setPanelRef = (node) => {
    localPanelRef.current = node;
    if (typeof panelRef === "function") {
      panelRef(node);
    } else if (panelRef) {
      panelRef.current = node;
    }
  };

  if (!open || !coords) return null;

  return createPortal(
    <div
      ref={setPanelRef}
      className={`history-dropdown history-dropdown-floating${coords.openUp ? " open-up" : ""}`}
      style={{ top: coords.top, left: coords.left }}
      role="menu"
    >
      {children}
    </div>,
    document.body,
  );
}
