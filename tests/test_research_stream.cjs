const { test } = require('node:test');
const assert = require('node:assert/strict');
const { consume } = require('../static/js/research-stream.js');

function stream(text, size = 1) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({ start(controller) {
    for (let i = 0; i < bytes.length; i += size) controller.enqueue(bytes.slice(i, i + size));
    controller.close();
  }});
}

test('handles split UTF-8, CRLF boundaries, comments and multiple events', async () => {
  const events = [];
  await consume(stream(': heartbeat\r\n\r\ndata: {"event":"status","data":{"message":"Café 📚"}}\r\n\r\ndata: {"event":"report","data":{"id":"rep_1"}}\r\n\r\n'), e => events.push(e));
  assert.equal(events.length, 2);
  assert.equal(events[0].data.message, 'Café 📚');
  assert.equal(events[1].data.id, 'rep_1');
});

test('accepts a final event without a trailing separator and multiline data', async () => {
  const events = [];
  await consume(stream('data: {"event":"report",\ndata: "data":{"id":"rep_last"}}', 7), e => events.push(e));
  assert.equal(events[0].data.id, 'rep_last');
});

test('rejects an interrupted stream instead of showing success', async () => {
  await assert.rejects(consume(stream('data: {"event":"status","data":{}}\n\n'), () => {}), /before a report/);
});

test('surfaces server failures and malformed events', async () => {
  await assert.rejects(consume(stream('data: {"event":"error","data":{"message":"Source unavailable"}}\n\n'), () => {}), /Source unavailable/);
  await assert.rejects(consume(stream('data: {invalid}\n\n'), () => {}), /unreadable response/);
});

test('propagates cancellation and releases the reader lock', async () => {
  let controller;
  const body = new ReadableStream({ start(c) { controller = c; } });
  const result = consume(body, () => {});
  controller.error(new DOMException('Stopped', 'AbortError'));
  await assert.rejects(result, { name: 'AbortError' });
  assert.equal(body.locked, false);
});

test('cancels the stream when rendering fails without swallowing the error', async () => {
  let cancelled = false;
  const body = new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode('data: {"event":"report","data":{}}\n\n')); },
    cancel() { cancelled = true; }
  });
  await assert.rejects(consume(body, () => { throw new Error('Render failed'); }), /Render failed/);
  assert.equal(cancelled, true);
  assert.equal(body.locked, false);
});
