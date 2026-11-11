/* Incremental SSE decoding shared by the browser and dependency-free Node tests. */
(function (root) {
  async function consume(body, onEvent) {
    if (!body) throw new Error("The server returned no response stream. Please retry.");
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let receivedReport = false;
    let finished = false;
    function dispatch(frame) {
      const data = frame.split(/\r?\n/).filter(line => line.startsWith("data:"))
        .map(line => line.slice(5).replace(/^ /, "")).join("\n");
      if (!data) return;
      let event;
      try { event = JSON.parse(data); }
      catch { throw new Error("The server sent an unreadable response. Please retry."); }
      if (event.event === "error") throw new Error(event.data?.message || "Research failed. Please retry.");
      onEvent(event);
      if (event.event === "report") receivedReport = true;
    }
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let separator;
        while ((separator = /\r?\n\r?\n/.exec(buffer))) {
          dispatch(buffer.slice(0, separator.index));
          buffer = buffer.slice(separator.index + separator[0].length);
        }
        if (done) {
          finished = true;
          if (buffer.trim()) dispatch(buffer);
          break;
        }
      }
      if (!receivedReport) throw new Error("The connection ended before a report arrived. Please retry.");
    } finally {
      if (!finished) await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }
  root.ResearchStream = { consume };
  if (typeof module !== "undefined") module.exports = root.ResearchStream;
})(globalThis);
