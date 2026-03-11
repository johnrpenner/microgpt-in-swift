#!/usr/bin/env swift
// MicroGPT in Swift v4.0
// Original port by John Roland Penner
// February 2026
//
// v4.0: BPE Tokenization — February 2026
//       - BPE tokenizer: 4096 subword tokens trained on corpus
//       - nEmbd: 64 -> 128  (larger embedding for larger vocab)
//       - blockSize: 80 chars -> 128 tokens (~320 chars of context)
//       - vocabSize: 97 chars -> 4096 BPE tokens
//       - Model file format: // microgpt4 header + VOCAB/END_VOCAB + BPE_MERGES/END_BPE blocks
//       - GPU default (--gpu flag no longer needed; CPU available via --cpu)
//       - Value{} autograd UNTOUCHED as CPU ground truth
//       - All v3.1 GPU infrastructure carried forward verbatim

// Andrej Karpathy is the Goat!

/**
 The most atomic way to train and inference a GPT in pure Swift.
 This file is the complete algorithm. Everything else is just efficiency.

 v4.0 Model File Format:
   // microgpt4
   4096
   VOCAB
   the
	and
   ing
   ...4096 lines, one token per line...
   END_VOCAB
   BPE_MERGES
   t h
   th e
   ...one merge pair per line (space-separated)...
   END_BPE
   layer0.attn_wk
   -0.054221...,  (weights)
*/

import Foundation
import Metal
import Darwin

// MARK: - FancyPrint
// by john roland penner / February 18, 2026
// Isolated output class — microgpt logic only calls FancyPrint.out() or Swift print().

let fancyPrintEnabled: Bool = true

class FancyPrint {
	static let cursorVisible:   Bool   = false
	static let spinnerFrames           = ["|", "/", "-", "\\"]
	static let spinnerDuration: Double = 0.1
	static let secondsPer80:    Double = 1.5

	static func hideCursor() { print("\u{001B}[?25l", terminator: ""); fflush(stdout) }
	static func showCursor()  { print("\u{001B}[?25h", terminator: ""); fflush(stdout) }

	// Get terminal width via ioctl — falls back to 80 if unavailable
	static func terminalWidth() -> Int {
		var w = winsize()
		if ioctl(STDOUT_FILENO, UInt(TIOCGWINSZ), &w) == 0 && w.ws_col > 0 {
			return Int(w.ws_col)
		}
		return 80
	}

	static func out(_ str: String, secondsPer80: Double = FancyPrint.secondsPer80) {
		if !fancyPrintEnabled { print(str); return }

		let leading  = String(str.prefix(while: { $0 == "\n" }))
		let coreStr  = String(str.drop(while: { $0 == "\n" })
							  .reversed().drop(while: { $0 == "\n" }).reversed())
		let trailingCount = str.count - leading.count - coreStr.count
		let trailing = trailingCount > 0 ? String(repeating: "\n", count: trailingCount) : ""

		if !leading.isEmpty { print(leading, terminator: "") }
		guard !coreStr.isEmpty else { print(trailing, terminator: ""); return }

		let cols = terminalWidth()

		if coreStr.count < cols {
			// ── SHORT PATH: fits on one line — classic \r spinner animation ───
			if !cursorVisible { hideCursor() }
			defer { if !cursorVisible { showCursor() } }

			let chars        = Array(coreStr)
			var pos          = 0
			var frameIdx     = 0
			var lastSpin     = DispatchTime.now()
			var lastChar     = DispatchTime.now()
			let charInterval = secondsPer80 / 80.0

			func redraw() {
				let revealed = String(chars[0..<pos])
				let spin     = spinnerFrames[frameIdx]
				let prefix   = pos == 0 ? "\n" : ""
				print("\(prefix)\r\u{001B}[K\(revealed)\(spin)", terminator: "")
				fflush(stdout)
			}

			redraw()
			while pos < chars.count {
				let now         = DispatchTime.now()
				let elapsedChar = Double(now.uptimeNanoseconds - lastChar.uptimeNanoseconds) / 1_000_000_000
				let elapsedSpin = Double(now.uptimeNanoseconds - lastSpin.uptimeNanoseconds) / 1_000_000_000
				var needsRedraw = false
				if elapsedChar >= charInterval { pos += 1; lastChar = now; needsRedraw = true }
				if elapsedSpin >= spinnerDuration {
					frameIdx = (frameIdx + 1) % spinnerFrames.count
					lastSpin = now; needsRedraw = true
				}
				if needsRedraw { redraw() }
				usleep(1_000)
			}
			print("\u{0008} ", terminator: "")
			print("")

		} else {
			// ── LONG PATH: wraps terminal — forward-only streaming reveal ─────
			// Never uses \r or cursor-up. Safe for any output length.
			if !cursorVisible { hideCursor() }
			defer { if !cursorVisible { showCursor() } }

			let chars        = Array(coreStr)
			var pos          = 0
			var frameIdx     = 0
			var lastSpin     = DispatchTime.now()
			var lastChar     = DispatchTime.now()
			let charInterval = secondsPer80 / 80.0

			// Opening newline + initial spinner
			print("\n\(spinnerFrames[frameIdx])", terminator: "")
			fflush(stdout)

			while pos < chars.count {
				let now         = DispatchTime.now()
				let elapsedChar = Double(now.uptimeNanoseconds - lastChar.uptimeNanoseconds) / 1_000_000_000
				let elapsedSpin = Double(now.uptimeNanoseconds - lastSpin.uptimeNanoseconds) / 1_000_000_000

				if elapsedChar >= charInterval {
					print("\u{0008}\(chars[pos])\(spinnerFrames[frameIdx])", terminator: "")
					fflush(stdout)
					pos += 1; lastChar = now
				} else if elapsedSpin >= spinnerDuration {
					frameIdx = (frameIdx + 1) % spinnerFrames.count
					print("\u{0008}\(spinnerFrames[frameIdx])", terminator: "")
					fflush(stdout)
					lastSpin = now
				}
				usleep(1_000)
			}
			print("\u{0008} ")  // erase final spinner, end line
		}

		if !trailing.isEmpty { print(trailing, terminator: "") }
	}
}

// MARK: - BPE Tokenizer

class BPETokenizer {
	var vocab:  [String]         = []   // index -> token string
	var tokToId: [String: Int]   = [:]  // token string -> index
	var merges: [(String, String)] = [] // ordered merge rules (left, right)

	// Special tokens
	var bosId: Int { vocab.count }      // BOS = one past last real token (set after build)
	private var _bosId: Int = 0
	var BOS: Int { _bosId }

	// ── Training ─────────────────────────────────────────────────────────────
	// Train BPE on a corpus of strings. targetVocabSize includes base characters.
	func train(corpus: [String], targetVocabSize: Int) {
		print("BPE: building base vocabulary from corpus...")

		// 1. Collect all unique characters as the base vocabulary
		var charSet = Set<Character>()
		for doc in corpus { charSet.formUnion(doc) }
		let baseChars = charSet.sorted().map { String($0) }
		vocab = baseChars
		for (i, t) in vocab.enumerated() { tokToId[t] = i }

		// 2. Represent each document as a sequence of current token indices
		//    We work with word-level splits to make BPE tractable:
		//    split on whitespace boundaries, keeping spaces as separate tokens.
		print("BPE: tokenizing corpus into word pieces...")
		var wordFreqs: [String: Int] = [:]
		for doc in corpus {
			// Split into "words" — runs of non-space chars + individual spaces
			var word = ""
			for ch in doc {
				if ch == " " {
					if !word.isEmpty { wordFreqs[word, default: 0] += 1; word = "" }
					wordFreqs[" ", default: 0] += 1
				} else {
					word.append(ch)
				}
			}
			if !word.isEmpty { wordFreqs[word, default: 0] += 1 }
		}

		// Represent each word as array of current token strings (initially chars)
		// workCorpus: [(token_sequence, frequency)]
		var workCorpus: [([String], Int)] = wordFreqs.map { (Array($0.key.map { String($0) }), $0.value) }

		print("BPE: merging pairs (target vocab: \(targetVocabSize))...")
		let targetMerges = targetVocabSize - vocab.count
		var mergesCount = 0

		while vocab.count < targetVocabSize {
			// Count all adjacent pairs weighted by frequency
			var pairCounts: [String: Int] = [:]
			for (tokens, freq) in workCorpus {
				for i in 0..<(tokens.count - 1) {
					let key = tokens[i] + "\u{0000}" + tokens[i+1]
					pairCounts[key, default: 0] += freq
				}
			}
			guard let bestKey = pairCounts.max(by: { $0.value < $1.value })?.key else { break }

			let parts = bestKey.split(separator: "\u{0000}", maxSplits: 1, omittingEmptySubsequences: false)
			guard parts.count == 2 else { break }
			let left  = String(parts[0])
			let right = String(parts[1])
			let merged = left + right

			merges.append((left, right))
			vocab.append(merged)
			tokToId[merged] = vocab.count - 1

			// Apply merge to work corpus
			workCorpus = workCorpus.map { (tokens, freq) in
				var newTokens: [String] = []
				var i = 0
				while i < tokens.count {
					if i < tokens.count - 1 && tokens[i] == left && tokens[i+1] == right {
						newTokens.append(merged)
						i += 2
					} else {
						newTokens.append(tokens[i])
						i += 1
					}
				}
				return (newTokens, freq)
			}

			mergesCount += 1
			if mergesCount % 500 == 0 {
				let pct = Int(100 * Double(mergesCount) / Double(targetMerges))
				print("  BPE: \(vocab.count) tokens (\(pct)%)...")
			}
		}

		_bosId = vocab.count
		print("BPE: vocabulary complete — \(vocab.count) tokens, \(merges.count) merges, BOS=\(_bosId)")
	}

	// ── Encoding ─────────────────────────────────────────────────────────────
	func encode(_ text: String) -> [Int] {
		// Start with character-level tokenization
		var tokens: [String] = text.map { String($0) }

		// Apply merges in order
		for (left, right) in merges {
			var i = 0
			var newTokens: [String] = []
			while i < tokens.count {
				if i < tokens.count - 1 && tokens[i] == left && tokens[i+1] == right {
					newTokens.append(left + right)
					i += 2
				} else {
					newTokens.append(tokens[i])
					i += 1
				}
			}
			tokens = newTokens
		}

		// Map to IDs, skip unknown tokens
		return tokens.compactMap { tokToId[$0] }
	}

	// ── Decoding ─────────────────────────────────────────────────────────────
	func decode(_ ids: [Int]) -> String {
		ids.compactMap { $0 < vocab.count ? vocab[$0] : nil }.joined()
	}

	// ── Model file serialization ──────────────────────────────────────────────
	func serialize() -> String {
		var out = "VOCAB\n"
		for token in vocab {
			let escaped = token
				.replacingOccurrences(of: "\\", with: "\\\\")
				.replacingOccurrences(of: "\n", with: "\\n")
				.replacingOccurrences(of: "\r", with: "\\r")
				.replacingOccurrences(of: "\t", with: "\\t")
			out += escaped + "\n"
		}
		out += "END_VOCAB\n"
		out += "BPE_MERGES\n"
		for (left, right) in merges {
			let el = left.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\t", with: "\\t")
			let er = right.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\t", with: "\\t")
			out += "\(el)\t\(er)\n"   // tab-separated pair — tab never appears in BPE tokens
		}
		out += "END_BPE\n"
		return out
	}

	// ── Model file deserialization ────────────────────────────────────────────
	func load(from lines: [String], startIdx: Int) -> Int {
		var idx = startIdx
		// Expect VOCAB
		guard idx < lines.count, lines[idx] == "VOCAB" else {
			print("BPE load error: expected VOCAB at line \(idx)"); return idx
		}
		idx += 1
		vocab = []; tokToId = [:]
		while idx < lines.count && lines[idx] != "END_VOCAB" {
			let escaped = lines[idx]
			let token = escaped
				.replacingOccurrences(of: "\\n", with: "\n")
				.replacingOccurrences(of: "\\r", with: "\r")
				.replacingOccurrences(of: "\\t", with: "\t")
				.replacingOccurrences(of: "\\\\", with: "\\")
			vocab.append(token)
			tokToId[token] = vocab.count - 1
			idx += 1
		}
		idx += 1 // skip END_VOCAB
		guard idx < lines.count, lines[idx] == "BPE_MERGES" else {
			print("BPE load error: expected BPE_MERGES at line \(idx)"); return idx
		}
		idx += 1
		merges = []
		while idx < lines.count && lines[idx] != "END_BPE" {
			// Tab-separated pair — tab never appears in BPE tokens
			let line = lines[idx]
			if let tabIdx = line.firstIndex(of: "\t") {
				var left  = String(line[line.startIndex..<tabIdx])
				var right = String(line[line.index(after: tabIdx)...])
				left  = left.replacingOccurrences(of: "\\n", with: "\n").replacingOccurrences(of: "\\t", with: "\t").replacingOccurrences(of: "\\\\", with: "\\")
				right = right.replacingOccurrences(of: "\\n", with: "\n").replacingOccurrences(of: "\\t", with: "\t").replacingOccurrences(of: "\\\\", with: "\\")
				merges.append((left, right))
			}
			idx += 1
		}
		idx += 1 // skip END_BPE
		_bosId = vocab.count
		return idx
	}
}

// MARK: - Autograd Value class (UNTOUCHED from v3.1)

class Value {
	var data: Double
	var grad: Double = 0.0
	private var children: [Value] = []
	private var localGrads: [Double] = []

	init(_ data: Double, children: [Value] = [], localGrads: [Double] = []) {
		self.data = data
		self.children = children
		self.localGrads = localGrads
	}

	static func + (lhs: Value, rhs: Value) -> Value {
		Value(lhs.data + rhs.data, children: [lhs, rhs], localGrads: [1.0, 1.0])
	}
	static func + (lhs: Value, rhs: Double) -> Value {
		Value(lhs.data + rhs, children: [lhs], localGrads: [1.0])
	}
	static func * (lhs: Value, rhs: Value) -> Value {
		Value(lhs.data * rhs.data, children: [lhs, rhs], localGrads: [rhs.data, lhs.data])
	}
	static func * (lhs: Value, rhs: Double) -> Value {
		Value(lhs.data * rhs, children: [lhs], localGrads: [rhs])
	}
	static func / (lhs: Value, rhs: Double) -> Value { lhs * (1.0 / rhs) }
	static func - (lhs: Value, rhs: Value) -> Value { lhs + (rhs * -1.0) }
	static prefix func - (v: Value) -> Value { v * -1.0 }

	func pow(_ exp: Double) -> Value {
		Value(Foundation.pow(data, exp), children: [self], localGrads: [exp * Foundation.pow(data, exp - 1)])
	}
	func relu() -> Value {
		Value(max(0, data), children: [self], localGrads: [data > 0 ? 1.0 : 0.0])
	}
	func log() -> Value {
		Value(Foundation.log(max(data, 1e-10)), children: [self], localGrads: [1.0 / max(data, 1e-10)])
	}
	func exp() -> Value {
		let e = Foundation.exp(data)
		return Value(e, children: [self], localGrads: [e])
	}

	func backward() {
		grad = 1.0
		var visited = Set<ObjectIdentifier>()
		var topo: [Value] = []
		func build(_ v: Value) {
			if visited.insert(ObjectIdentifier(v)).inserted {
				v.children.forEach { build($0) }
				topo.append(v)
			}
		}
		build(self)
		for v in topo.reversed() {
			for (child, lg) in zip(v.children, v.localGrads) { child.grad += v.grad * lg }
		}
	}
}

// MARK: - Math utilities (UNTOUCHED from v3.1)

func softmax(_ xs: [Value]) -> [Value] {
	let m = xs.map { $0.data }.max() ?? 0
	let exps = xs.map { ($0 + (-m)).exp() }
	let s = exps.reduce(Value(0), +)
	return exps.map { $0 / s.data }
}

func rmsnorm(_ x: [Value]) -> [Value] {
	let n = Double(x.count)
	let ms = x.map { $0 * $0 }.reduce(Value(0), +) / n
	let scale = (ms + 1e-5).pow(-0.5)
	return x.map { $0 * scale }
}

func linear(_ x: [Value], _ w: [[Value]]) -> [Value] {
	w.map { row in zip(row, x).map(*).reduce(Value(0), +) }
}

func matrix(nout: Int, nin: Int) -> [[Value]] {
	(0..<nout).map { _ in
		(0..<nin).map { _ in Value(Double.random(in: -0.2...0.2)) }
	}
}

// MARK: - Metal Shaders (UNTOUCHED from v3.1)

let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

// ─── FORWARD KERNELS ──────────────────────────────────────────────────────────

kernel void matVecMul(
	device const float* x      [[ buffer(0) ]],
	device const float* W      [[ buffer(1) ]],
	device       float* out    [[ buffer(2) ]],
	device const int&   nin    [[ buffer(3) ]],
	device const int&   nout   [[ buffer(4) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if (gid >= uint(nout)) return;
	float sum = 0.0f;
	for (int j = 0; j < nin; j++) sum += W[gid * nin + j] * x[j];
	out[gid] = sum;
}

kernel void softmaxForward(
	device const float* x    [[ buffer(0) ]],
	device       float* out  [[ buffer(1) ]],
	device const int&   n    [[ buffer(2) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if (gid != 0) return;
	float maxVal = x[0];
	for (int i = 1; i < n; i++) if (x[i] > maxVal) maxVal = x[i];
	float sum = 0.0f;
	for (int i = 0; i < n; i++) { out[i] = exp(x[i] - maxVal); sum += out[i]; }
	for (int i = 0; i < n; i++) out[i] /= sum;
}

kernel void rmsNormForward(
	device const float* x    [[ buffer(0) ]],
	device       float* out  [[ buffer(1) ]],
	device const int&   n    [[ buffer(2) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if (gid != 0) return;
	float ms = 0.0f;
	for (int i = 0; i < n; i++) ms += x[i] * x[i];
	ms /= n;
	float scale = rsqrt(ms + 1e-5f);
	for (int i = 0; i < n; i++) out[i] = x[i] * scale;
}

kernel void reluForward(
	device const float* x   [[ buffer(0) ]],
	device       float* out [[ buffer(1) ]],
	uint gid [[ thread_position_in_grid ]])
{
	out[gid] = max(0.0f, x[gid]);
}

// ─── BACKWARD KERNELS ─────────────────────────────────────────────────────────

kernel void matVecMulBackwardW(
	device const float* dOut [[ buffer(0) ]],
	device const float* x    [[ buffer(1) ]],
	device       float* dW   [[ buffer(2) ]],
	device const int&   nin  [[ buffer(3) ]],
	device const int&   nout [[ buffer(4) ]],
	uint gid [[ thread_position_in_grid ]])
{
	int i = gid / nin;
	int j = gid % nin;
	if (i >= nout || j >= nin) return;
	dW[i * nin + j] += dOut[i] * x[j];
}

kernel void matVecMulBackwardX(
	device const float* dOut [[ buffer(0) ]],
	device const float* W    [[ buffer(1) ]],
	device       float* dx   [[ buffer(2) ]],
	device const int&   nin  [[ buffer(3) ]],
	device const int&   nout [[ buffer(4) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if (gid >= uint(nin)) return;
	float sum = 0.0f;
	for (int i = 0; i < nout; i++) sum += W[i * nin + gid] * dOut[i];
	dx[gid] += sum;
}

kernel void softmaxCEBackward(
	device const float* probs  [[ buffer(0) ]],
	device       float* dLogits[[ buffer(1) ]],
	device const int&   target [[ buffer(2) ]],
	device const int&   n      [[ buffer(3) ]],
	device const float& scale  [[ buffer(4) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if (gid >= uint(n)) return;
	float p = probs[gid];
	dLogits[gid] = (p - (int(gid) == target ? 1.0f : 0.0f)) * scale;
}

kernel void rmsNormBackward(
	device const float* x    [[ buffer(0) ]],
	device const float* dOut [[ buffer(1) ]],
	device       float* dx   [[ buffer(2) ]],
	device const int&   n    [[ buffer(3) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if (gid != 0) return;
	float ms = 0.0f;
	for (int i = 0; i < n; i++) ms += x[i] * x[i];
	ms /= n;
	float scale = rsqrt(ms + 1e-5f);
	float dot = 0.0f;
	for (int i = 0; i < n; i++) dot += dOut[i] * x[i] * scale;
	for (int i = 0; i < n; i++) dx[i] = scale * (dOut[i] - x[i] * scale * dot / n);
}

kernel void reluBackward(
	device const float* pre  [[ buffer(0) ]],
	device const float* dOut [[ buffer(1) ]],
	device       float* dx   [[ buffer(2) ]],
	uint gid [[ thread_position_in_grid ]])
{
	dx[gid] = pre[gid] > 0.0f ? dOut[gid] : 0.0f;
}

// ─── UTILITY KERNELS ──────────────────────────────────────────────────────────

kernel void addInPlace(
	device float* dst [[ buffer(0) ]],
	device const float* src [[ buffer(1) ]],
	uint gid [[ thread_position_in_grid ]])
{
	dst[gid] += src[gid];
}

kernel void zeroBuffer(
	device float* buf [[ buffer(0) ]],
	uint gid [[ thread_position_in_grid ]])
{
	buf[gid] = 0.0f;
}

kernel void copyBuffer(
	device const float* src [[ buffer(0) ]],
	device       float* dst [[ buffer(1) ]],
	uint gid [[ thread_position_in_grid ]])
{
	dst[gid] = src[gid];
}

// ─── ADAM UPDATE ──────────────────────────────────────────────────────────────

kernel void adamUpdate(
	device       float* params [[ buffer(0) ]],
	device const float* grads  [[ buffer(1) ]],
	device       float* m      [[ buffer(2) ]],
	device       float* v      [[ buffer(3) ]],
	device const float& lr     [[ buffer(4) ]],
	device const float& beta1  [[ buffer(5) ]],
	device const float& beta2  [[ buffer(6) ]],
	device const float& eps    [[ buffer(7) ]],
	device const float& beta1t [[ buffer(8) ]],
	device const float& beta2t [[ buffer(9) ]],
	uint gid [[ thread_position_in_grid ]])
{
	float g  = grads[gid];
	m[gid]   = beta1 * m[gid] + (1.0f - beta1) * g;
	v[gid]   = beta2 * v[gid] + (1.0f - beta2) * g * g;
	float mH = m[gid] / (1.0f - beta1t);
	float vH = v[gid] / (1.0f - beta2t);
	params[gid] -= lr * mH / (sqrt(vH) + eps);
}
"""

// MARK: - MetalContext (UNTOUCHED from v3.1)

class MetalContext {
	var gpuAvailable: Bool = false
	var device: MTLDevice?
	var commandQueue: MTLCommandQueue?
	var library: MTLLibrary?
	var pipelines: [String: MTLComputePipelineState] = [:]

	init() {
		guard let dev = MTLCreateSystemDefaultDevice() else {
			print("Metal: [No GPU device — falling back to CPU]"); return
		}
		guard let queue = dev.makeCommandQueue() else {
			print("Metal: [Could not create command queue — falling back to CPU]"); return
		}
		do {
			let lib = try dev.makeLibrary(source: metalShaderSource, options: nil)
			let names = [
				"matVecMul", "softmaxForward", "rmsNormForward", "reluForward",
				"matVecMulBackwardW", "matVecMulBackwardX",
				"softmaxCEBackward", "rmsNormBackward", "reluBackward",
				"addInPlace", "zeroBuffer", "copyBuffer", "adamUpdate"
			]
			for name in names {
				guard let fn = lib.makeFunction(name: name),
					  let ps = try? dev.makeComputePipelineState(function: fn) else {
					print("Metal: [Failed to compile kernel \(name) — falling back to CPU]"); return
				}
				pipelines[name] = ps
			}
			device = dev; commandQueue = queue; library = lib
			gpuAvailable = true
			print("Metal: [GPU initialized: \(dev.name)]")
		} catch {
			print("Metal: [Shader compile error — falling back to CPU]: \(error)")
		}
	}

	func makeBuffer(_ data: [Float]) -> MTLBuffer? {
		device?.makeBuffer(bytes: data, length: data.count * 4, options: .storageModeShared)
	}
	func makeBuffer(count: Int) -> MTLBuffer? {
		device?.makeBuffer(length: max(count, 1) * 4, options: .storageModeShared)
	}
	func readBuffer(_ buf: MTLBuffer, count: Int) -> [Float] {
		let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
		return Array(UnsafeBufferPointer(start: ptr, count: count))
	}

	func dispatch(kernel: String, buffers: [MTLBuffer], threadCount: Int) {
		guard threadCount > 0,
			  let queue = commandQueue,
			  let pipeline = pipelines[kernel],
			  let cmdBuf = queue.makeCommandBuffer(),
			  let encoder = cmdBuf.makeComputeCommandEncoder() else { return }
		encoder.setComputePipelineState(pipeline)
		for (i, buf) in buffers.enumerated() { encoder.setBuffer(buf, offset: 0, index: i) }
		let tpg = min(pipeline.maxTotalThreadsPerThreadgroup, threadCount)
		encoder.dispatchThreadgroups(
			MTLSize(width: (threadCount + tpg - 1) / tpg, height: 1, depth: 1),
			threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
		encoder.endEncoding()
		cmdBuf.commit(); cmdBuf.waitUntilCompleted()
	}
}

let metal = MetalContext()

// ── CPU fallback wrappers (used in validation mode, UNTOUCHED from v3.1) ─────

func linearGPU(_ xF: [Double], _ wF: [[Double]], nin: Int, nout: Int) -> [Double] {
	guard metal.gpuAvailable,
		  let xB = metal.makeBuffer(xF.map { Float($0) }),
		  let wB = metal.makeBuffer(wF.flatMap { $0.map { Float($0) } }),
		  let oB = metal.makeBuffer(count: nout)
	else { return wF.map { row in zip(row, xF).map(*).reduce(0, +) } }
	var ninI = Int32(nin); var noutI = Int32(nout)
	guard let ninBuf  = metal.device?.makeBuffer(bytes: &ninI,  length: 4, options: .storageModeShared),
		  let noutBuf = metal.device?.makeBuffer(bytes: &noutI, length: 4, options: .storageModeShared)
	else { return wF.map { row in zip(row, xF).map(*).reduce(0, +) } }
	metal.dispatch(kernel: "matVecMul", buffers: [xB, wB, oB, ninBuf, noutBuf], threadCount: nout)
	return metal.readBuffer(oB, count: nout).map { Double($0) }
}

func softmaxGPU(_ logits: [Double]) -> [Double] {
	let n = logits.count
	guard let lB = metal.makeBuffer(logits.map { Float($0) }),
		  let pB = metal.makeBuffer(count: n) else {
		let m = logits.max()!; let e = logits.map { Foundation.exp($0 - m) }
		let s = e.reduce(0, +); return e.map { $0 / s }
	}
	var nI = Int32(n)
	guard let nBuf = metal.device?.makeBuffer(bytes: &nI, length: 4, options: .storageModeShared) else {
		let m = logits.max()!; let e = logits.map { Foundation.exp($0 - m) }
		let s = e.reduce(0, +); return e.map { $0 / s }
	}
	metal.dispatch(kernel: "softmaxForward", buffers: [lB, pB, nBuf], threadCount: n)
	return metal.readBuffer(pB, count: n).map { Double($0) }
}

func rmsnormGPU(_ xF: [Double]) -> [Double] {
	let n = xF.count
	guard let xB = metal.makeBuffer(xF.map { Float($0) }),
		  let oB = metal.makeBuffer(count: n) else {
		let ms = xF.map { $0*$0 }.reduce(0, +) / Double(n)
		let s  = 1.0 / sqrt(ms + 1e-5)
		return xF.map { $0 * s }
	}
	var nI = Int32(n)
	guard let nBuf = metal.device?.makeBuffer(bytes: &nI, length: 4, options: .storageModeShared) else {
		let ms = xF.map { $0*$0 }.reduce(0, +) / Double(n)
		let s  = 1.0 / sqrt(ms + 1e-5)
		return xF.map { $0 * s }
	}
	metal.dispatch(kernel: "rmsNormForward", buffers: [xB, oB, nBuf], threadCount: n)
	return metal.readBuffer(oB, count: n).map { Double($0) }
}

// MARK: - GPUTrainer (adapted from v3.1 — only V, E, BS updated for v4 hyperparams)

class GPUTrainer {
	let E: Int; let H: Int; let L: Int; let BS: Int; let V: Int; let HD: Int

	var wte: MTLBuffer!; var wpe: MTLBuffer!; var lmHead: MTLBuffer!
	var lW: [[MTLBuffer]] = []
	var dWte: MTLBuffer!; var dWpe: MTLBuffer!; var dLmHead: MTLBuffer!
	var lG: [[MTLBuffer]] = []
	var mWte: MTLBuffer!; var vWte: MTLBuffer!
	var mWpe: MTLBuffer!; var vWpe: MTLBuffer!
	var mLmHead: MTLBuffer!; var vLmHead: MTLBuffer!
	var lM: [[MTLBuffer]] = []; var lV: [[MTLBuffer]] = []
	var lSizes: [Int] = []

	var cE: MTLBuffer!; var c4E: MTLBuffer!; var cV: MTLBuffer!
	var cEE: MTLBuffer!; var cScaleF: MTLBuffer!; var cTargetI: MTLBuffer!
	var cLr: MTLBuffer!; var cB1: MTLBuffer!; var cB2: MTLBuffer!
	var cEps: MTLBuffer!; var cB1t: MTLBuffer!; var cB2t: MTLBuffer!

	var sE1: MTLBuffer!; var sE2: MTLBuffer!; var sE3: MTLBuffer!
	var sE4: MTLBuffer!; var sE5: MTLBuffer!; var sE6: MTLBuffer!
	var sE7: MTLBuffer!; var sE8: MTLBuffer!; var sE9: MTLBuffer!
	var sE10: MTLBuffer!; var sE11: MTLBuffer!; var sE12: MTLBuffer!
	var s4E1: MTLBuffer!; var s4E2: MTLBuffer!; var s4E3: MTLBuffer!
	var sV1: MTLBuffer!

	var layerActBufs: [[MTLBuffer]] = []
	var layerMlpBufs: [[MTLBuffer]] = []
	var fwdX: MTLBuffer!; var fwdXPre0: MTLBuffer!; var fwdLogits: MTLBuffer!

	init(nEmbd: Int, nHead: Int, nLayer: Int, blockSize: Int, vocabSize: Int) {
		E = nEmbd; H = nHead; L = nLayer; BS = blockSize; V = vocabSize; HD = nEmbd / nHead
		lSizes = [E*E, E*E, E*E, E*E, 4*E*E, E*4*E]
		setupBuffers()
		setupActBuffers()
	}

	func newBuf(_ n: Int) -> MTLBuffer {
		guard let b = metal.makeBuffer(count: max(n, 1)) else {
			fatalError("GPUTrainer: out of Metal memory allocating \(n*4) bytes")
		}
		return b
	}
	func setInt(_ buf: MTLBuffer, _ val: Int)   { buf.contents().storeBytes(of: Int32(val),  as: Int32.self) }
	func setFloat(_ buf: MTLBuffer, _ val: Float) { buf.contents().storeBytes(of: val, as: Float.self) }

	func setupBuffers() {
		wte = newBuf(V*E); wpe = newBuf(BS*E); lmHead = newBuf(V*E)
		dWte = newBuf(V*E); dWpe = newBuf(BS*E); dLmHead = newBuf(V*E)
		mWte = newBuf(V*E); vWte = newBuf(V*E)
		mWpe = newBuf(BS*E); vWpe = newBuf(BS*E)
		mLmHead = newBuf(V*E); vLmHead = newBuf(V*E)
		for _ in 0..<L {
			lW.append(lSizes.map { newBuf($0) })
			lG.append(lSizes.map { newBuf($0) })
			lM.append(lSizes.map { newBuf($0) })
			lV.append(lSizes.map { newBuf($0) })
		}
		cE   = newBuf(1); setInt(cE,   E)
		c4E  = newBuf(1); setInt(c4E, 4*E)
		cV   = newBuf(1); setInt(cV,   V)
		cEE  = newBuf(1); setInt(cEE,  E)
		cScaleF  = newBuf(1); cTargetI = newBuf(1)
		cLr  = newBuf(1); cB1  = newBuf(1); cB2  = newBuf(1)
		cEps = newBuf(1); cB1t = newBuf(1); cB2t = newBuf(1)
		sE1 = newBuf(E); sE2 = newBuf(E); sE3 = newBuf(E)
		sE4 = newBuf(E); sE5 = newBuf(E); sE6 = newBuf(E)
		sE7 = newBuf(E); sE8 = newBuf(E); sE9 = newBuf(E)
		sE10 = newBuf(E); sE11 = newBuf(E); sE12 = newBuf(E)
		s4E1 = newBuf(4*E); s4E2 = newBuf(4*E); s4E3 = newBuf(4*E)
		sV1 = newBuf(V)
	}

	func setupActBuffers() {
		for _ in 0..<L {
			layerActBufs.append((0..<10).map { _ in newBuf(E) })
			layerMlpBufs.append([newBuf(4*E)])
		}
		fwdX = newBuf(E); fwdXPre0 = newBuf(E); fwdLogits = newBuf(V)
	}

	func syncFrom(_ sd: [String: [[Value]]]) {
		func write(_ buf: MTLBuffer, _ mat: [[Value]]) {
			let f = mat.flatMap { $0.map { Float($0.data) } }
			buf.contents().copyMemory(from: f, byteCount: f.count * 4)
		}
		write(wte, sd["wte"]!); write(wpe, sd["wpe"]!); write(lmHead, sd["lm_head"]!)
		for li in 0..<L {
			let keys = ["layer\(li).attn_wq","layer\(li).attn_wk","layer\(li).attn_wv",
						"layer\(li).attn_wo","layer\(li).mlp_fc1","layer\(li).mlp_fc2"]
			for (wi, key) in keys.enumerated() { write(lW[li][wi], sd[key]!) }
		}
	}

	func syncTo(_ sd: [String: [[Value]]]) {
		func read(_ buf: MTLBuffer, _ mat: [[Value]]) {
			let nr = mat.count; let nc = mat[0].count
			let f = metal.readBuffer(buf, count: nr * nc)
			for i in 0..<nr { for j in 0..<nc { mat[i][j].data = Double(f[i*nc+j]); mat[i][j].grad = 0 } }
		}
		read(wte, sd["wte"]!); read(wpe, sd["wpe"]!); read(lmHead, sd["lm_head"]!)
		for li in 0..<L {
			let keys = ["layer\(li).attn_wq","layer\(li).attn_wk","layer\(li).attn_wv",
						"layer\(li).attn_wo","layer\(li).mlp_fc1","layer\(li).mlp_fc2"]
			for (wi, key) in keys.enumerated() { read(lW[li][wi], sd[key]!) }
		}
	}

	func zeroGrads() {
		let all: [MTLBuffer] = [dWte, dWpe, dLmHead] + lG.flatMap { $0 }
		for b in all { metal.dispatch(kernel: "zeroBuffer", buffers: [b], threadCount: b.length/4) }
	}

	// ── Primitive ops ──────────────────────────────────────────────────────────
	func linFwd(_ x: MTLBuffer, _ w: MTLBuffer, nin: Int, nout: Int, out: MTLBuffer) {
		let ninB:  MTLBuffer = (nin  == E) ? cE : c4E
		let noutB: MTLBuffer = (nout == E) ? cE : (nout == 4*E ? c4E : cV)
		metal.dispatch(kernel: "matVecMul", buffers: [x, w, out, ninB, noutB], threadCount: nout)
	}
	func smFwd(_ x: MTLBuffer, n: Int, out: MTLBuffer) {
		let nB: MTLBuffer = (n == V) ? cV : cE
		metal.dispatch(kernel: "softmaxForward", buffers: [x, out, nB], threadCount: n)
	}
	func rnFwd(_ x: MTLBuffer, n: Int, out: MTLBuffer) {
		metal.dispatch(kernel: "rmsNormForward", buffers: [x, out, cE], threadCount: n)
	}
	func reluFwd(_ x: MTLBuffer, n: Int, out: MTLBuffer) {
		metal.dispatch(kernel: "reluForward", buffers: [x, out], threadCount: n)
	}
	func linBwd(dOut: MTLBuffer, x: MTLBuffer, w: MTLBuffer, dW: MTLBuffer,
				nin: Int, nout: Int, dx: MTLBuffer) {
		let ninB:  MTLBuffer = (nin  == E) ? cE : c4E
		let noutB: MTLBuffer = (nout == E) ? cE : (nout == 4*E ? c4E : cV)
		metal.dispatch(kernel: "matVecMulBackwardW", buffers: [dOut, x, dW, ninB, noutB], threadCount: nout*nin)
		metal.dispatch(kernel: "matVecMulBackwardX", buffers: [dOut, w, dx, ninB, noutB], threadCount: nin)
	}
	func rnBwd(x: MTLBuffer, dOut: MTLBuffer, n: Int, dx: MTLBuffer) {
		metal.dispatch(kernel: "rmsNormBackward", buffers: [x, dOut, dx, cE], threadCount: n)
	}
	func reluBwd(pre: MTLBuffer, dOut: MTLBuffer, n: Int, dx: MTLBuffer) {
		metal.dispatch(kernel: "reluBackward", buffers: [pre, dOut, dx], threadCount: n)
	}
	func addIP(_ dst: MTLBuffer, _ src: MTLBuffer, n: Int) {
		metal.dispatch(kernel: "addInPlace", buffers: [dst, src], threadCount: n)
	}
	func copyBuf(_ src: MTLBuffer, _ dst: MTLBuffer, n: Int) {
		metal.dispatch(kernel: "copyBuffer", buffers: [src, dst], threadCount: n)
	}
	func zeroBuf(_ buf: MTLBuffer, n: Int) {
		metal.dispatch(kernel: "zeroBuffer", buffers: [buf], threadCount: n)
	}

	func embLookup(_ table: MTLBuffer, row: Int, dim: Int, into dst: MTLBuffer) {
		dst.contents().copyMemory(from: table.contents().advanced(by: row * dim * 4), byteCount: dim * 4)
	}
	func embBwd(_ dTable: MTLBuffer, _ dVec: MTLBuffer, row: Int, dim: Int) {
		let src = dVec.contents().bindMemory(to: Float.self, capacity: dim)
		let dst = dTable.contents().bindMemory(to: Float.self, capacity: dTable.length/4)
		for j in 0..<dim { dst[row * dim + j] += src[j] }
	}

	struct LayerActs {
		var xIn: MTLBuffer; var xNorm1: MTLBuffer; var q: MTLBuffer
		var attnW: [[Float]]
		var xAttn: MTLBuffer; var xNorm2: MTLBuffer
		var preRelu: MTLBuffer; var xAfterAttn: MTLBuffer
	}

	func forward(tokenId: Int, posId: Int,
				 kCache: inout [[MTLBuffer]], vCache: inout [[MTLBuffer]],
				 acts: inout [LayerActs]) {
		embLookup(wte, row: tokenId, dim: E, into: sE1)
		embLookup(wpe, row: posId,   dim: E, into: sE2)
		copyBuf(sE1, fwdXPre0, n: E); addIP(fwdXPre0, sE2, n: E)
		rnFwd(fwdXPre0, n: E, out: fwdX)

		for li in 0..<L {
			let ab = layerActBufs[li]; let mb = layerMlpBufs[li]
			copyBuf(fwdX, ab[0], n: E)
			rnFwd(fwdX, n: E, out: ab[1])
			linFwd(ab[1], lW[li][0], nin: E, nout: E, out: ab[2])
			linFwd(ab[1], lW[li][1], nin: E, nout: E, out: ab[3])
			linFwd(ab[1], lW[li][2], nin: E, nout: E, out: ab[4])
			kCache[li].append(ab[3]); vCache[li].append(ab[4])

			let qF = metal.readBuffer(ab[2], count: E)
			var attnWeightsAllHeads: [[Float]] = []
			var xAttnF = [Float](repeating: 0, count: E)
			for h in 0..<H {
				let hs = h * HD
				let qH = Array(qF[hs..<hs+HD])
				var logitsA = [Float]()
				for t in 0..<kCache[li].count {
					let kF = metal.readBuffer(kCache[li][t], count: E)
					let dot = zip(qH, kF[hs..<hs+HD]).map(*).reduce(0, +)
					logitsA.append(dot / sqrt(Float(HD)))
				}
				let maxA = logitsA.max()!
				var aw = logitsA.map { Foundation.exp($0 - maxA) }
				let sumA = aw.reduce(0, +); aw = aw.map { $0 / sumA }
				attnWeightsAllHeads.append(aw)
				for t in 0..<vCache[li].count {
					let vF = metal.readBuffer(vCache[li][t], count: E)
					for j in 0..<HD { xAttnF[hs+j] += aw[t] * vF[hs+j] }
				}
			}
			ab[5].contents().copyMemory(from: xAttnF, byteCount: E * 4)
			linFwd(ab[5], lW[li][3], nin: E, nout: E, out: ab[6])
			copyBuf(ab[6], ab[7], n: E); addIP(ab[7], ab[0], n: E)
			rnFwd(ab[7], n: E, out: ab[8])
			linFwd(ab[8], lW[li][4], nin: E, nout: 4*E, out: mb[0])
			reluFwd(mb[0], n: 4*E, out: s4E1)
			linFwd(s4E1, lW[li][5], nin: 4*E, nout: E, out: ab[9])
			copyBuf(ab[9], fwdX, n: E); addIP(fwdX, ab[7], n: E)

			acts.append(LayerActs(xIn: ab[0], xNorm1: ab[1], q: ab[2],
								  attnW: attnWeightsAllHeads, xAttn: ab[5],
								  xNorm2: ab[8], preRelu: mb[0], xAfterAttn: ab[7]))
		}
		linFwd(fwdX, lmHead, nin: E, nout: V, out: fwdLogits)
	}

	func backward(posId: Int, tokenId: Int, acts: [LayerActs],
				  kCache: [[MTLBuffer]], vCache: [[MTLBuffer]]) {
		zeroBuf(sE1, n: E)
		linBwd(dOut: sV1, x: fwdX, w: lmHead, dW: dLmHead, nin: E, nout: V, dx: sE1)

		for li in stride(from: L-1, through: 0, by: -1) {
			let a = acts[li]
			copyBuf(sE1, sE2, n: E)
			reluFwd(a.preRelu, n: 4*E, out: s4E1)
			zeroBuf(s4E2, n: 4*E)
			linBwd(dOut: sE1, x: s4E1, w: lW[li][5], dW: lG[li][5], nin: 4*E, nout: E, dx: s4E2)
			zeroBuf(s4E3, n: 4*E)
			reluBwd(pre: a.preRelu, dOut: s4E2, n: 4*E, dx: s4E3)
			zeroBuf(sE3, n: E)
			linBwd(dOut: s4E3, x: a.xNorm2, w: lW[li][4], dW: lG[li][4], nin: E, nout: 4*E, dx: sE3)
			zeroBuf(sE4, n: E)
			rnBwd(x: a.xAfterAttn, dOut: sE3, n: E, dx: sE4)
			addIP(sE2, sE4, n: E); copyBuf(sE2, sE1, n: E)

			copyBuf(sE1, sE2, n: E)
			zeroBuf(sE3, n: E)
			linBwd(dOut: sE1, x: a.xAttn, w: lW[li][3], dW: lG[li][3], nin: E, nout: E, dx: sE3)

			let dxAttnF = metal.readBuffer(sE3, count: E)
			var dqF = [Float](repeating: 0, count: E)
			var dkF = [Float](repeating: 0, count: E)
			var dvF = [Float](repeating: 0, count: E)
			for h in 0..<H {
				let hs = h * HD; let aw = a.attnW[h]; let seqLen = aw.count
				let dAttnOut = Array(dxAttnF[hs..<hs+HD])
				for t in 0..<seqLen { for j in 0..<HD { dvF[hs+j] += aw[t] * dAttnOut[j] } }
				var dAW = [Float](repeating: 0, count: seqLen)
				for t in 0..<seqLen {
					let vF = metal.readBuffer(vCache[li][t], count: E)
					for j in 0..<HD { dAW[t] += dAttnOut[j] * vF[hs+j] }
				}
				let dotAWdAW = zip(aw, dAW).map(*).reduce(0, +)
				let scale = 1.0 / sqrt(Float(HD))
				var dAttnLogits = [Float](repeating: 0, count: seqLen)
				for t in 0..<seqLen { dAttnLogits[t] = aw[t] * (dAW[t] - dotAWdAW) }
				let qF = metal.readBuffer(a.q, count: E)
				for t in 0..<seqLen {
					let kF = metal.readBuffer(kCache[li][t], count: E)
					let g = dAttnLogits[t] * scale
					for j in 0..<HD { dqF[hs+j] += g * kF[hs+j]; dkF[hs+j] += g * qF[hs+j] }
				}
			}
			sE4.contents().copyMemory(from: dqF, byteCount: E * 4)
			sE5.contents().copyMemory(from: dkF, byteCount: E * 4)
			sE6.contents().copyMemory(from: dvF, byteCount: E * 4)
			zeroBuf(sE7, n: E); zeroBuf(sE8, n: E); zeroBuf(sE9, n: E)
			linBwd(dOut: sE4, x: a.xNorm1, w: lW[li][0], dW: lG[li][0], nin: E, nout: E, dx: sE7)
			linBwd(dOut: sE5, x: a.xNorm1, w: lW[li][1], dW: lG[li][1], nin: E, nout: E, dx: sE8)
			linBwd(dOut: sE6, x: a.xNorm1, w: lW[li][2], dW: lG[li][2], nin: E, nout: E, dx: sE9)
			addIP(sE7, sE8, n: E); addIP(sE7, sE9, n: E)
			zeroBuf(sE8, n: E)
			rnBwd(x: a.xIn, dOut: sE7, n: E, dx: sE8)
			addIP(sE2, sE8, n: E); copyBuf(sE2, sE1, n: E)
		}
		zeroBuf(sE2, n: E)
		rnBwd(x: fwdXPre0, dOut: sE1, n: E, dx: sE2)
		embBwd(dWte, sE2, row: tokenId, dim: E)
		embBwd(dWpe, sE2, row: posId,   dim: E)
	}

	func adamStep(step: Int, lr: Float, beta1: Float, beta2: Float, eps: Float) {
		let b1t = Float(Foundation.pow(Double(beta1), Double(step)))
		let b2t = Float(Foundation.pow(Double(beta2), Double(step)))
		setFloat(cLr, lr); setFloat(cB1, beta1); setFloat(cB2, beta2)
		setFloat(cEps, eps); setFloat(cB1t, b1t); setFloat(cB2t, b2t)
		let sc: [MTLBuffer] = [cLr, cB1, cB2, cEps, cB1t, cB2t]
		func upd(_ p: MTLBuffer, _ g: MTLBuffer, _ m: MTLBuffer, _ v: MTLBuffer, _ n: Int) {
			metal.dispatch(kernel: "adamUpdate", buffers: [p, g, m, v] + sc, threadCount: n)
		}
		upd(wte, dWte, mWte, vWte, V*E)
		upd(wpe, dWpe, mWpe, vWpe, BS*E)
		upd(lmHead, dLmHead, mLmHead, vLmHead, V*E)
		for li in 0..<L { for wi in 0..<6 { upd(lW[li][wi], lG[li][wi], lM[li][wi], lV[li][wi], lSizes[wi]) } }
	}
}

// MARK: - GPT Model (UNTOUCHED from v3.1 — works with any token IDs)

var stateDict: [String: [[Value]]] = [:]

func gpt(tokenId: Int, posId: Int, keys: inout [[[Value]]], values: inout [[[Value]]],
		 nEmbd: Int, nHead: Int, nLayer: Int) -> [Value] {
	let headDim = nEmbd / nHead
	let tokEmb = stateDict["wte"]![tokenId]
	let posEmb = stateDict["wpe"]![posId]
	var x = zip(tokEmb, posEmb).map(+)
	x = rmsnorm(x)
	for li in 0..<nLayer {
		let xResidual = x
		x = rmsnorm(x)
		let q = linear(x, stateDict["layer\(li).attn_wq"]!)
		let k = linear(x, stateDict["layer\(li).attn_wk"]!)
		let v = linear(x, stateDict["layer\(li).attn_wv"]!)
		keys[li].append(k); values[li].append(v)
		var xAttn: [Value] = []
		for h in 0..<nHead {
			let hs = h * headDim
			let qH = Array(q[hs..<hs+headDim])
			let kH = keys[li].map { Array($0[hs..<hs+headDim]) }
			let vH = values[li].map { Array($0[hs..<hs+headDim]) }
			let attnLogits = kH.map { kt in zip(qH, kt).map(*).reduce(Value(0), +) / sqrt(Double(headDim)) }
			let attnWeights = softmax(attnLogits)
			let headOut = (0..<headDim).map { j in
				zip(attnWeights, vH).map { w, vt in w * vt[j] }.reduce(Value(0), +)
			}
			xAttn.append(contentsOf: headOut)
		}
		x = linear(xAttn, stateDict["layer\(li).attn_wo"]!)
		x = zip(x, xResidual).map(+)
		let xResidual2 = x
		x = rmsnorm(x)
		x = linear(x, stateDict["layer\(li).mlp_fc1"]!)
		x = x.map { $0.relu() }
		x = linear(x, stateDict["layer\(li).mlp_fc2"]!)
		x = zip(x, xResidual2).map(+)
	}
	return linear(x, stateDict["lm_head"]!)
}

// MARK: - Utilities

let validationTolerance: Double = 0.0001

func validateLoss(step: Int, cpuLoss: Double, gpuLoss: Double) -> Bool {
	let diff = abs(cpuLoss - gpuLoss)
	let passed = diff < validationTolerance
	print(String(format: "  [Validate] Step %3d | CPU: %.6f | GPU: %.6f | diff: %.6f | %@",
				step, cpuLoss, gpuLoss, diff, passed ? "✓ PASS" : "✗ FAIL"))
	return passed
}
func formattedTime(_ d: Date) -> String {
	let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd HH:mm:ss"; return f.string(from: d)
}
func formattedElapsed(_ s: Double) -> String {
	let h = Int(s)/3600; let m = (Int(s)%3600)/60; let sec = Int(s)%60
	return h > 0 ? "\(h)h \(m)m \(sec)s" : m > 0 ? "\(m)m \(sec)s" : "\(sec)s"
}
func paramCount() -> Int { stateDict.values.flatMap { $0.flatMap { $0 } }.count }
func paramStr(_ n: Int) -> String {
	n >= 1_000_000 ? String(format: "%.2fM", Double(n) / 1_000_000)
				   : String(format: "%d,%03d", n / 1000, n % 1000)
}

// MARK: - Main Program

print("\n--| MicroGPT 👾 Swift Edition v4 |---")

let args = CommandLine.arguments
var mode = "train"
var inputFiles: [String] = []
var modelPath  = "model.txt"
var numSteps   = 1000
var temperature = 0.5
var maxSamples  = 100000
var useGPU      = true    // GPU is DEFAULT in v4; use --cpu to override
var validateMode = false

var i = 1
while i < args.count {
	switch args[i] {
	case "--train":    mode = "train"
	case "--generate": mode = "generate"
	case "--model":    i += 1; if i < args.count { modelPath = args[i] }
	case "--steps":    i += 1; if i < args.count { numSteps = Int(args[i]) ?? 1000 }
	case "--temperature", "--temp": i += 1; if i < args.count { temperature = Double(args[i]) ?? 0.5 }
	case "--samples":  i += 1; if i < args.count { maxSamples = Int(args[i]) ?? 100000 }
	case "--cpu":      useGPU = false
	case "--validate": validateMode = true
	case "--help", "-h": break
	default: if args[i].hasSuffix(".txt") { inputFiles.append(args[i]) }
	}
	i += 1
}

// v4 Model hyperparameters
let nEmbd     = 128    // up from 64 — needed for 4096-token vocab
let nHead     = 4
let nLayer    = 4
let blockSize = 128    // tokens (≈320 chars of context at avg 2.5 chars/token)
let targetBPEVocab = 4096

// ── Model version identifier ────────────────────────────────────────────────
let modelVersion = "// microgpt4"

if mode == "train" {

	let trainStart = Date()
	print("Training started: \(formattedTime(trainStart))")

	// ── Load corpus ──────────────────────────────────────────────────────────
	var docs: [String] = []
	if inputFiles.isEmpty {
		print("No input files — using example data...")
		docs = ["hello world", "swift is great", "machine learning is fun"]
	} else {
		print("Loading files...")
		for file in inputFiles {
			let paths = [file,
						 FileManager.default.currentDirectoryPath + "/" + file,
						 (file as NSString).expandingTildeInPath]
			var loaded = false
			for path in paths {
				if let content = try? String(contentsOfFile: path, encoding: .utf8) {
					let lines = content.components(separatedBy: .newlines)
						.map { $0.trimmingCharacters(in: .whitespaces) }
						.filter { !$0.isEmpty && $0.count <= 400 }
					docs.append(contentsOf: lines)
					print("  ✓ Loaded \(file): \(lines.count) lines")
					loaded = true; break
				}
			}
			if !loaded { print("  ✗ Could not read: \(file)") }
		}
	}
	if docs.isEmpty { print("❌ No documents loaded."); exit(1) }
	if docs.count > maxSamples {
		print("\nDataset is large (\(docs.count) lines). Sampling \(maxSamples) lines for training...")
		docs = Array(docs.shuffled().prefix(maxSamples))
	}
	docs.shuffle()
	print("Loaded \(docs.count) documents")

	// ── Train BPE tokenizer ──────────────────────────────────────────────────
	let tokenizer = BPETokenizer()
	tokenizer.train(corpus: docs, targetVocabSize: targetBPEVocab)
	let vocabSize = tokenizer.BOS + 1   // BOS is one past the last real token
	let BOS = tokenizer.BOS

	print("Vocabulary size: \(vocabSize) (BPE tokens + BOS)")

	// ── Training mode reporting ──────────────────────────────────────────────
	if useGPU && !metal.gpuAvailable {
		print("Training mode: CPU (Metal unavailable)")
		useGPU = false
	} else if useGPU {
		print("Training mode: GPU — full forward + backward + Adam on Metal")
	} else {
		print("Training mode: CPU (original Value{} autograd)")
	}
	if validateMode { print("Validation mode: ON") }

	// ── Initialize weights ───────────────────────────────────────────────────
	stateDict["wte"]     = matrix(nout: vocabSize, nin: nEmbd)
	stateDict["wpe"]     = matrix(nout: blockSize, nin: nEmbd)
	stateDict["lm_head"] = matrix(nout: vocabSize, nin: nEmbd)
	for li in 0..<nLayer {
		stateDict["layer\(li).attn_wq"] = matrix(nout: nEmbd, nin: nEmbd)
		stateDict["layer\(li).attn_wk"] = matrix(nout: nEmbd, nin: nEmbd)
		stateDict["layer\(li).attn_wv"] = matrix(nout: nEmbd, nin: nEmbd)
		stateDict["layer\(li).attn_wo"] = matrix(nout: nEmbd, nin: nEmbd)
		stateDict["layer\(li).mlp_fc1"] = matrix(nout: 4 * nEmbd, nin: nEmbd)
		stateDict["layer\(li).mlp_fc2"] = matrix(nout: nEmbd, nin: 4 * nEmbd)
	}

	let params = stateDict.values.flatMap { $0.flatMap { $0 } }
	print("Number of parameters: \(paramStr(params.count))\n")
	print("Hyperparams: nEmbd \(nEmbd), nHead \(nHead), nLayers \(nLayer), blockSize \(blockSize)")

	// ── GPU trainer ──────────────────────────────────────────────────────────
	var trainer: GPUTrainer? = nil
	if useGPU && metal.gpuAvailable {
		trainer = GPUTrainer(nEmbd: nEmbd, nHead: nHead, nLayer: nLayer,
							 blockSize: blockSize, vocabSize: vocabSize)
	}

	var gpuSnapshot: [String: [[Double]]] = [:]
	if validateMode {
		for (key, mat) in stateDict { gpuSnapshot[key] = mat.map { $0.map { $0.data } } }
	}

	var adamM = [Double](repeating: 0.0, count: params.count)
	var adamV = [Double](repeating: 0.0, count: params.count)
	//let learningRate = 0.01
	let learningRate = 0.003
	let beta1 = 0.85; let beta2 = 0.99; let epsAdam = 1e-8

	var valPasses = 0; var valFails = 0

	print("Training for \(numSteps) steps...")
	let loopStart = Date()

	for step in 0..<numSteps {
		let doc = docs[step % docs.count]

		// Encode document with BPE
		var tokens = [BOS]
		tokens.append(contentsOf: tokenizer.encode(doc))
		tokens.append(BOS)
		// Clamp to blockSize+1 (need pairs for next-token prediction)
		if tokens.count > blockSize + 1 { tokens = Array(tokens.prefix(blockSize + 1)) }
		let seqLen = min(blockSize, tokens.count - 1)
		guard seqLen > 0 else { continue }

		var stepLoss = 0.0

		if let tr = trainer, useGPU && !validateMode {
			// ── GPU PATH ─────────────────────────────────────────────────────
			tr.syncFrom(stateDict); tr.zeroGrads()
			var kCache = [[MTLBuffer]](repeating: [], count: nLayer)
			var vCache = [[MTLBuffer]](repeating: [], count: nLayer)
			var totalLoss: Float = 0

			for posId in 0..<seqLen {
				let tokenId  = tokens[posId]
				let targetId = tokens[posId + 1]
				var posActs: [GPUTrainer.LayerActs] = []
				tr.forward(tokenId: tokenId, posId: posId,
						   kCache: &kCache, vCache: &vCache, acts: &posActs)
				let logitsF = metal.readBuffer(tr.fwdLogits, count: vocabSize)
				let maxL = logitsF.max()!
				var exps = logitsF.map { Foundation.exp(Double($0 - maxL)) }
				let sumE = exps.reduce(0, +); exps = exps.map { $0 / sumE }
				totalLoss += Float(-Foundation.log(max(exps[targetId], 1e-10)))
				tr.smFwd(tr.fwdLogits, n: vocabSize, out: tr.sV1)
				tr.setInt(tr.cTargetI, targetId)
				tr.setFloat(tr.cScaleF, 1.0 / Float(seqLen))
				metal.dispatch(kernel: "softmaxCEBackward",
							   buffers: [tr.sV1, tr.sV1, tr.cTargetI, tr.cV, tr.cScaleF],
							   threadCount: vocabSize)
				tr.backward(posId: posId, tokenId: tokenId,
							acts: posActs, kCache: kCache, vCache: vCache)
			}
			stepLoss = Double(totalLoss / Float(seqLen))
			let lrT = Float(learningRate * (1.0 - Double(step) / Double(numSteps)))
			tr.adamStep(step: step + 1, lr: lrT, beta1: Float(beta1), beta2: Float(beta2), eps: Float(epsAdam))
			tr.syncTo(stateDict)

		} else {
			// ── CPU PATH ─────────────────────────────────────────────────────
			var keys   = [[[Value]]](repeating: [], count: nLayer)
			var values = [[[Value]]](repeating: [], count: nLayer)
			var losses: [Value] = []
			for posId in 0..<seqLen {
				let tokenId  = tokens[posId]; let targetId = tokens[posId + 1]
				let logits = gpt(tokenId: tokenId, posId: posId, keys: &keys, values: &values,
								 nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
				let probs = softmax(logits); losses.append(-probs[targetId].log())
			}
			let loss = losses.reduce(Value(0), +) / Double(seqLen)
			stepLoss = loss.data; loss.backward()
			let lrT = learningRate * (1.0 - Double(step) / Double(numSteps))
			for (idx, p) in params.enumerated() {
				adamM[idx] = beta1 * adamM[idx] + (1 - beta1) * p.grad
				adamV[idx] = beta2 * adamV[idx] + (1 - beta2) * p.grad * p.grad
				let mHat = adamM[idx] / (1 - Foundation.pow(beta1, Double(step + 1)))
				let vHat = adamV[idx] / (1 - Foundation.pow(beta2, Double(step + 1)))
				p.data -= lrT * mHat / (sqrt(vHat) + epsAdam)
				p.grad = 0.0
			}
		}

		// ── Validation ───────────────────────────────────────────────────────
		if validateMode {
			var gpuSD: [String: [[Value]]] = [:]
			for (key, mat) in gpuSnapshot { gpuSD[key] = mat.map { $0.map { Value($0) } } }
			let saved = stateDict; stateDict = gpuSD
			var gpuKeys = [[[Value]]](repeating: [], count: nLayer)
			var gpuVals = [[[Value]]](repeating: [], count: nLayer)
			var gpuLosses: [Value] = []
			for posId in 0..<seqLen {
				let tokenId = tokens[posId]; let targetId = tokens[posId + 1]
				let logits = gpt(tokenId: tokenId, posId: posId, keys: &gpuKeys, values: &gpuVals,
								 nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
				let probs = softmax(logits); gpuLosses.append(-probs[targetId].log())
			}
			let gpuLoss = gpuLosses.reduce(Value(0), +).data / Double(seqLen)
			stateDict = saved
			let passed = validateLoss(step: step + 1, cpuLoss: stepLoss, gpuLoss: gpuLoss)
			if passed { valPasses += 1 } else { valFails += 1 }
			for (key, mat) in stateDict { gpuSnapshot[key] = mat.map { $0.map { $0.data } } }
		}

		// ── Progress reporting ───────────────────────────────────────────────
		let elapsed = Date().timeIntervalSince(loopStart)
		let sps     = Double(step + 1) / elapsed
		let eta     = sps > 0 ? Double(numSteps - step - 1) / sps : 0
		if step == 0 || (step + 1) % 10 == 0 {
			print(String(format: "step %4d / %4d | loss %.4f | %.2f steps/s | ETA: %@",
						step + 1, numSteps, stepLoss, sps, formattedElapsed(eta)))
		}
	}

	let trainEnd = Date()
	let totalElapsed = trainEnd.timeIntervalSince(trainStart)
	let avgSPS = Double(numSteps) / totalElapsed

	print("\n--- Training Complete ----------")
	print("  Started:   \(formattedTime(trainStart))")
	print("  Finished:  \(formattedTime(trainEnd))")
	print("  Elapsed:   \(formattedElapsed(totalElapsed))")
	print(String(format: "  Avg speed: %.3f steps/s", avgSPS))
	if validateMode {
		print("  Validation: \(valPasses) PASS / \(valFails) FAIL out of \(numSteps) steps")
		print(valFails == 0
			? "  Result: ✓ GPU matches CPU within tolerance (\(validationTolerance))"
			: "  Result: ✗ GPU diverged on \(valFails) steps")
	}

	// ── Save model ───────────────────────────────────────────────────────────
	print("  Saving model to \(modelPath)...")
	var modelData = "\(modelVersion)\n"
	modelData += "\(vocabSize)\n"
	modelData += tokenizer.serialize()
	for (key, mat) in stateDict.sorted(by: { $0.key < $1.key }) {
		modelData += "\(key)\n"
		for row in mat { modelData += row.map { String($0.data) }.joined(separator: ",") + "\n" }
	}
	try? modelData.write(toFile: modelPath, atomically: true, encoding: .utf8)
	print("  Model saved!")

} else if mode == "generate" {

	// ── Load model ───────────────────────────────────────────────────────────
	print("Loading model from \(modelPath)...")
	guard let modelContent = try? String(contentsOfFile: modelPath, encoding: .utf8) else {
		print("Error: Could not load model file '\(modelPath)'. Train a model first!"); exit(1)
	}

	let rawLines = modelContent.components(separatedBy: .newlines)
	var lineIdx = 0

	// Check version header
	guard rawLines[lineIdx].hasPrefix("// microgpt") else {
		print("Error: Model file missing version header. Is this a microgpt4 model?"); exit(1)
	}
	let fileVersion = rawLines[lineIdx]
	if fileVersion != modelVersion {
		print("Warning: Model version '\(fileVersion)' != expected '\(modelVersion)'")
		print("         File may have been trained with a different version.")
	}
	lineIdx += 1

	let vocabSize = Int(rawLines[lineIdx])!; lineIdx += 1
	print("Vocabulary size: \(vocabSize)")

	// Load BPE tokenizer
	let tokenizer = BPETokenizer()
	// Filter empty lines only where needed — BPE loader handles its own line-by-line
	lineIdx = tokenizer.load(from: rawLines, startIdx: lineIdx)
	let BOS = tokenizer.BOS
	print("BPE: loaded \(tokenizer.vocab.count) tokens, \(tokenizer.merges.count) merges")

	// Load weights
	let matrixSizes: [String: (Int, Int)] = {
		var s: [String: (Int, Int)] = [
			"wte": (vocabSize, nEmbd), "wpe": (blockSize, nEmbd), "lm_head": (vocabSize, nEmbd)
		]
		for li in 0..<nLayer {
			s["layer\(li).attn_wq"] = (nEmbd, nEmbd); s["layer\(li).attn_wk"] = (nEmbd, nEmbd)
			s["layer\(li).attn_wv"] = (nEmbd, nEmbd); s["layer\(li).attn_wo"] = (nEmbd, nEmbd)
			s["layer\(li).mlp_fc1"] = (4*nEmbd, nEmbd); s["layer\(li).mlp_fc2"] = (nEmbd, 4*nEmbd)
		}
		return s
	}()

	let lines = rawLines.filter { !$0.isEmpty }
	// Reindex from where BPE loader left off — find first weight key
	var weightLineIdx = 0
	for (li, line) in lines.enumerated() {
		if matrixSizes[line] != nil { weightLineIdx = li; break }
	}
	var wli = weightLineIdx
	while wli < lines.count {
		let key = lines[wli]; wli += 1
		guard let (nrows, _) = matrixSizes[key] else { continue }
		var mat: [[Value]] = []
		for _ in 0..<nrows {
			if wli >= lines.count { break }
			mat.append(lines[wli].split(separator: ",").map { Value(Double($0)!) })
			wli += 1
		}
		if mat.count == nrows { stateDict[key] = mat }
	}
	print("Model loaded!\n")

	let pc = paramCount()
	print("═══════════════════════════════════════")
	print("  Interactive Generation Mode")
	print("═══════════════════════════════════════")
	print("  Parameters: \(paramStr(pc)) | Temperature: \(temperature)")
	print("  Hyperparams: nEmbd \(nEmbd), nHead \(nHead), nLayers \(nLayer), blockSize \(blockSize)")
	print("───────────────────────────────────────")
	print("  Type your prompt and press Enter.")
	print("  Commands: quit  |  multi <n>  |  temp <n>")
	print("═══════════════════════════════════════\n")

	func generate(from promptText: String, numSamples: Int = 1) {
		for sampleIdx in 0..<numSamples {
			var keys   = [[[Value]]](repeating: [], count: nLayer)
			var values = [[[Value]]](repeating: [], count: nLayer)
			var position = 0

			// Encode prompt into BPE tokens and feed into model
			let promptTokens = tokenizer.encode(promptText)
			for tokenId in promptTokens {
				if position >= blockSize { break }
				_ = gpt(tokenId: tokenId, posId: position, keys: &keys, values: &values,
						nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
				position += 1
			}

			// Generate new tokens
			var generatedIds: [Int] = []
			var tokenId = BOS
			for posId in position..<blockSize {
				let logits = gpt(tokenId: tokenId, posId: posId, keys: &keys, values: &values,
								 nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
				let scaled = logits.map { $0 / temperature }
				let probs  = softmax(scaled)
				let weights = probs.map { $0.data }
				let total   = weights.reduce(0, +)
				var rand    = Double.random(in: 0..<total)
				tokenId = 0
				for (idx, w) in weights.enumerated() { rand -= w; if rand <= 0 { tokenId = idx; break } }
				if tokenId == BOS { break }
				generatedIds.append(tokenId)
			}

			let generated = tokenizer.decode(generatedIds)
			if numSamples > 1 {
				FancyPrint.out(String(format: "  %d: %@%@", sampleIdx + 1, promptText, generated))
			} else {
				FancyPrint.out("\(promptText)\(generated)")
			}
		}
	}

	var numSamples = 1
	while true {
		print("> ", terminator: ""); fflush(stdout)
		guard let input = readLine() else { break }
		let trimmed = input.trimmingCharacters(in: .whitespaces)
		if trimmed.isEmpty { continue }
		if trimmed.lowercased() == "exit" || trimmed.lowercased() == "quit" {
			FancyPrint.out("\nGoodbye! 👋"); break
		}
		if trimmed.lowercased().hasPrefix("multi ") {
			if let n = Int(trimmed.dropFirst(6).trimmingCharacters(in: .whitespaces)) {
				numSamples = max(1, min(n, 20)); print("✓ Will generate \(numSamples) variations\n")
			} else { print("Usage: multi <number>\n") }
			continue
		}
		if trimmed.lowercased().hasPrefix("temp ") {
			if let t = Double(trimmed.dropFirst(5).trimmingCharacters(in: .whitespaces)) {
				temperature = max(0.1, min(t, 2.0)); print("✓ Temperature: \(temperature)\n")
			} else { print("Usage: temp <number>\n") }
			continue
		}
		generate(from: trimmed, numSamples: numSamples); print("")
		if numSamples > 1 { numSamples = 1 }
	}
}

// Usage shown only with --help or no args
if args.contains("--help") || args.contains("-h") || args.count == 1 {
	print("Usage:")
	print("  ./microgpt4 --train file.txt [--steps 1000] [--model model.txt]")
	print("  ./microgpt4 --train file.txt --cpu  [--steps 1000]   (force CPU)")
	print("  ./microgpt4 --train file.txt --validate [--steps 10]")
	print("  ./microgpt4 --generate [--temp 0.5] [--model model.txt]")
	print("")
	print("Options:")
	print("  --steps N     Training steps (default: 1000)")
	print("  --samples N   Max corpus lines (default: 100000)")
	print("  --temp N      Generation temperature 0.1-2.0 (default: 0.5)")
	print("  --model path  Model file path (default: model.txt)")
	print("  --cpu         Force CPU training (default: GPU if available)")
	print("  --validate    Compare CPU vs GPU loss each step")
	print("  --help        Show this help")
}
