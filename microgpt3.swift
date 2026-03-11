#!/usr/bin/env swift
// MicroGPT in Swift v3.1
// Original port by John Roland Penner
// February 15, 2026
//
// v3.0: Metal GPU Acceleration — February 2026
//       Milestone 1: GPU forward pass validation confirmed bit-identical to CPU (diff=0.000000)
//       Milestone 2: GPU backward pass — explicit gradient kernels for all ops
//                    Full forward + backward + Adam on GPU
//                    Value{} class retained UNTOUCHED as CPU ground truth
//
// Bug fixed (v3.0 initial): --gpu incorrectly fed GPU-returned Values (no autograd children)
//            into CPU backward(), producing zero gradients and corrupt weights.
//            Fix: --gpu now uses CPU forward+backward until Milestone 2 GPU backward is active.
//            Validation (--validate) was always correct — unaffected by this bug.

// Andrej Karpathy is the Goat!

/**
 The most atomic way to train and inference a GPT in pure Swift.
 This file is the complete algorithm.
 Everything else is just efficiency.
 
 Based on @karpathy's micrograd GPT
 Multi-threaded version for performance
 
 v3.0 GPU Strategy (Option A — per-position sequential):
 - The Value{} scalar autograd system is UNTOUCHED and remains the CPU ground truth
 - A parallel Metal GPU path implements explicit forward + backward tensor kernels
 - Forward: GPU wrappers for linear, softmax, rmsnorm, relu (validated bit-identical to CPU)
 - Backward: explicit MSL gradient kernels mirror Value.backward() chain rule exactly
 - Intermediate activations saved during GPU forward pass for use in GPU backward
 - Validation mode (--validate): CPU and GPU forward run in parallel, losses compared
 - If Metal device unavailable, silently falls back to CPU
 - Double->Float32->Double precision validated bit-identical at 6 decimal places on M1
 */

import Foundation
import Metal
import Darwin

// MARK: - FancyPrint
// by john roland penner / February 18, 2026
// Isolated output class — microgpt logic only calls FancyPrint.out() or Swift print().
// Toggle with fancyPrintEnabled to switch between animated and plain output.

let fancyPrintEnabled: Bool = true

class FancyPrint {
	
	// Controls cursor visibility during animated reveal
	// false → hide real cursor (cleaner look, spinner is the only "cursor")
	// true  → keep real cursor visible (spinner appears to "lead" across the line)
	static let cursorVisible: Bool = false
	
	static let spinnerFrames    = ["|", "/", "-", "\\"]
	static let spinnerDuration  = 0.1    // seconds per spinner frame
	static let secondsPer80     = 1.5    // seconds to reveal 80 characters
	
	static func hideCursor() {
		print("\u{001B}[?25l", terminator: ""); fflush(stdout)
	}
	static func showCursor() {
		print("\u{001B}[?25h", terminator: ""); fflush(stdout)
	}
	
	// Main entry point — drop-in replacement for print() in generation output.
	// If fancyPrintEnabled is false this is identical to Swift print().
	static func out(_ str: String, secondsPer80: Double = FancyPrint.secondsPer80) {
		if !fancyPrintEnabled {
			print(str); return
		}
		
		// Strip leading/trailing newlines — animate only the core text.
		let leading  = String(str.prefix(while: { $0 == "\n" }))
		let coreStr  = String(str.drop(while: { $0 == "\n" })
							  .reversed().drop(while: { $0 == "\n" }).reversed())
		let trailingCount = str.count - leading.count - coreStr.count
		let trailing = trailingCount > 0 ? String(repeating: "\n", count: trailingCount) : ""
		
		if !leading.isEmpty { print(leading, terminator: "") }
		
		guard !coreStr.isEmpty else {
			print(trailing, terminator: "")
			return
		}
		
		if !cursorVisible { hideCursor() }
		defer { if !cursorVisible { showCursor() } }
		
		let chars        = Array(coreStr)
		var pos          = 0
		var frameIdx     = 0
		var lastSpin     = DispatchTime.now()
		var lastChar     = DispatchTime.now()
		let charInterval = secondsPer80 / 80.0
		
		// Animate on the CURRENT line — no \n before starting.
		// Each redraw: \r to line start, \u{001B}[K to clear to end, then revealed+spinner.
		// This stays on one line as long as output < terminal width.
		// When output >= 80 chars the terminal wraps naturally; we don't fight it.
		func redraw() {
			let revealed = String(chars[0..<pos])
			let spin     = spinnerFrames[frameIdx]
			print("\r\u{001B}[K\(revealed)\(spin)", terminator: "")
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
		
		// Animation done. Every character is already on screen correctly.
		// Just erase the spinner (one char) by overwriting with a space, then newline.
		// Never reprint coreStr — that's what causes the double output.
		print("\u{0008} ", terminator: "")  // backspace over spinner, overwrite with space
		print("")                            // newline to end the line cleanly
		
		if !trailing.isEmpty { print(trailing, terminator: "") }
	}
}



// Let there be Autograd, to recursively apply the chain rule through a computation graph
class Value {
	var data: Double
	var grad: Double = 0.0
	private var children: [Value] = []
	private var localGrads: [Double] = []
	
	init(_ data: Double, children: [Value] = [], localGrads: [Double] = []) {
		self.data = data            // scalar value of this node calculated during forward pass
		self.children = children    // the child nodes this node depends on
		self.localGrads = localGrads // local derivative of this node w.r.t. its children
	}
	
	static func + (lhs: Value, rhs: Value) -> Value {
		return Value(lhs.data + rhs.data, children: [lhs, rhs], localGrads: [1.0, 1.0])
	}
	
	static func + (lhs: Value, rhs: Double) -> Value {
		return lhs + Value(rhs)
	}
	
	static func * (lhs: Value, rhs: Value) -> Value {
		return Value(lhs.data * rhs.data, children: [lhs, rhs], localGrads: [rhs.data, lhs.data])
	}
	
	static func * (lhs: Value, rhs: Double) -> Value {
		return lhs * Value(rhs)
	}
	
	static func * (lhs: Double, rhs: Value) -> Value {
		return Value(lhs) * rhs
	}
	
	static prefix func - (value: Value) -> Value {
		return value * -1.0
	}
	
	static func - (lhs: Value, rhs: Value) -> Value {
		return lhs + (-rhs)
	}
	
	static func - (lhs: Value, rhs: Double) -> Value {
		return lhs + (-rhs)
	}
	
	static func / (lhs: Value, rhs: Value) -> Value {
		return lhs * rhs.pow(-1.0)
	}
	
	static func / (lhs: Value, rhs: Double) -> Value {
		return lhs * Value(rhs).pow(-1.0)
	}
	
	func pow(_ exponent: Double) -> Value {
		return Value(Foundation.pow(data, exponent),
					children: [self],
					localGrads: [exponent * Foundation.pow(data, exponent - 1)])
	}
	
	func log() -> Value {
		return Value(Foundation.log(data), children: [self], localGrads: [1.0 / data])
	}
	
	func exp() -> Value {
		let expData = Foundation.exp(data)
		return Value(expData, children: [self], localGrads: [expData])
	}
	
	func relu() -> Value {
		return Value(max(0, data), children: [self], localGrads: [data > 0 ? 1.0 : 0.0])
	}
	
	func backward() {
		var topo: [Value] = []
		var visited = Set<ObjectIdentifier>()
		
		func buildTopo(_ v: Value) {
			let id = ObjectIdentifier(v)
			if !visited.contains(id) {
				visited.insert(id)
				for child in v.children {
					buildTopo(child)
				}
				topo.append(v)
			}
		}
		
		buildTopo(self)
		self.grad = 1.0
		
		for v in topo.reversed() {
			for (child, localGrad) in zip(v.children, v.localGrads) {
				child.grad += localGrad * v.grad
			}
		}
	}
}

// MARK: - [ORIGINAL - untouched] Helper Functions

func gaussRandom(mean: Double = 0.0, std: Double = 1.0) -> Double {
	let u1 = Double.random(in: 0.0...1.0)
	let u2 = Double.random(in: 0.0...1.0)
	let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
	return mean + std * z0
}

func matrix(nout: Int, nin: Int, std: Double = 0.08) -> [[Value]] {
	return (0..<nout).map { _ in
		(0..<nin).map { _ in Value(gaussRandom(std: std)) }
	}
}

func linear(_ x: [Value], _ w: [[Value]]) -> [Value] {
	return w.map { wo in
		zip(wo, x).map(*).reduce(Value(0), +)
	}
}

// Multi-threaded version for larger matrices (used during training)
func linearParallel(_ x: [Value], _ w: [[Value]]) -> [Value] {
	let nout = w.count
	guard nout > 8 else { return linear(x, w) } // Use serial for small matrices
	
	var result = [Value](repeating: Value(0), count: nout)
	let queue = DispatchQueue(label: "linear", attributes: .concurrent)
	let group = DispatchGroup()
	
	for i in 0..<nout {
		queue.async(group: group) {
			let val = zip(w[i], x).map(*).reduce(Value(0), +)
			result[i] = val
		}
	}
	group.wait()
	return result
}

func softmax(_ logits: [Value]) -> [Value] {
	let maxVal = logits.map { $0.data }.max()!
	let exps = logits.map { ($0 - maxVal).exp() }
	let total = exps.reduce(Value(0), +)
	return exps.map { $0 / total }
}

func rmsnorm(_ x: [Value]) -> [Value] {
	let ms = x.map { $0 * $0 }.reduce(Value(0), +) / Double(x.count)
	let scale = (ms + 1e-5).pow(-0.5)
	return x.map { $0 * scale }
}

// MARK: - [NEW - Metal] MSL Shader Source

// All Metal compute kernels in one inline MSL string, compiled at runtime.
// Forward kernels: mirror CPU functions exactly (same algorithm, Float32).
// Backward kernels: implement closed-form gradients matching Value.backward() chain rule.
let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

// ─── FORWARD KERNELS ─────────────────────────────────────────────────────────

// matVecMul: mirrors linear() -- out[i] = sum_j(w[i*nin+j] * x[j])
kernel void matVecMul(
	device const float* x    [[ buffer(0) ]],
	device const float* w    [[ buffer(1) ]],
	device       float* out  [[ buffer(2) ]],
	constant     int&   nin  [[ buffer(3) ]],
	constant     int&   nout [[ buffer(4) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if ((int)gid >= nout) return;
	float sum = 0.0f;
	int row = (int)gid * nin;
	for (int j = 0; j < nin; j++) sum += w[row + j] * x[j];
	out[gid] = sum;
}

// softmaxForward: numerically stable softmax, mirrors softmax()
kernel void softmaxForward(
	device const float* logits [[ buffer(0) ]],
	device       float* probs  [[ buffer(1) ]],
	constant     int&   n      [[ buffer(2) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if ((int)gid >= n) return;
	float maxVal = logits[0];
	for (int i = 1; i < n; i++) if (logits[i] > maxVal) maxVal = logits[i];
	float sum = 0.0f;
	for (int i = 0; i < n; i++) sum += exp(logits[i] - maxVal);
	probs[gid] = exp(logits[gid] - maxVal) / sum;
}

// rmsNormForward: mirrors rmsnorm() -- scale = rsqrt(mean(x^2)+1e-5), out=x*scale
kernel void rmsNormForward(
	device const float* x    [[ buffer(0) ]],
	device       float* out  [[ buffer(1) ]],
	constant     int&   n    [[ buffer(2) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if ((int)gid >= n) return;
	float ms = 0.0f;
	for (int i = 0; i < n; i++) ms += x[i] * x[i];
	ms /= (float)n;
	out[gid] = x[gid] * rsqrt(ms + 1e-5f);
}

// reluForward: mirrors relu() -- max(0, x)
kernel void reluForward(
	device const float* x   [[ buffer(0) ]],
	device       float* out [[ buffer(1) ]],
	uint gid [[ thread_position_in_grid ]])
{
	out[gid] = max(0.0f, x[gid]);
}

// ─── BACKWARD KERNELS ────────────────────────────────────────────────────────
// Each kernel mirrors exactly what Value.backward() propagates via chain rule:
//   child.grad += localGrad * v.grad

// matVecMulBackwardW: dW[i*nin+j] += dOut[i] * x[j]   (outer product, accumulated)
kernel void matVecMulBackwardW(
	device const float* dOut [[ buffer(0) ]],
	device const float* x    [[ buffer(1) ]],
	device       float* dW   [[ buffer(2) ]],
	constant     int&   nin  [[ buffer(3) ]],
	constant     int&   nout [[ buffer(4) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if ((int)gid >= nout * nin) return;
	int i = (int)gid / nin;
	int j = (int)gid % nin;
	dW[gid] += dOut[i] * x[j];
}

// matVecMulBackwardX: dx[j] += sum_i(W[i*nin+j] * dOut[i])  (W^T @ dOut)
kernel void matVecMulBackwardX(
	device const float* dOut [[ buffer(0) ]],
	device const float* w    [[ buffer(1) ]],
	device       float* dx   [[ buffer(2) ]],
	constant     int&   nin  [[ buffer(3) ]],
	constant     int&   nout [[ buffer(4) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if ((int)gid >= nin) return;
	float sum = 0.0f;
	for (int i = 0; i < nout; i++) sum += w[i * nin + (int)gid] * dOut[i];
	dx[gid] += sum;
}

// softmaxCEBackward: combined softmax + cross-entropy backward.
// When loss = -log(probs[target])/seqLen:
//   dLogits[i] = (probs[i] - (i==target?1:0)) * scale   where scale=1/seqLen
// This is the analytic simplification of backprop through softmax + log + negate + sum.
kernel void softmaxCEBackward(
	device const float* probs   [[ buffer(0) ]],
	device       float* dLogits [[ buffer(1) ]],
	constant     int&   target  [[ buffer(2) ]],
	constant     int&   n       [[ buffer(3) ]],
	constant     float& scale   [[ buffer(4) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if ((int)gid >= n) return;
	float indicator = ((int)gid == target) ? 1.0f : 0.0f;
	dLogits[gid] = (probs[gid] - indicator) * scale;
}

// rmsNormBackward: gradient through rmsnorm.
// Forward: scale=rsqrt(mean(x^2)+eps), out=x*scale
// Backward: dx[j] = scale*(dOut[j] - x[j]*scale*dot(dOut,out)/n)
kernel void rmsNormBackward(
	device const float* x    [[ buffer(0) ]],
	device const float* dOut [[ buffer(1) ]],
	device       float* dx   [[ buffer(2) ]],
	constant     int&   n    [[ buffer(3) ]],
	uint gid [[ thread_position_in_grid ]])
{
	if ((int)gid >= n) return;
	float ms = 0.0f;
	for (int i = 0; i < n; i++) ms += x[i] * x[i];
	ms /= (float)n;
	float scale = rsqrt(ms + 1e-5f);
	float dotDoutOut = 0.0f;
	for (int i = 0; i < n; i++) dotDoutOut += dOut[i] * x[i] * scale;
	dx[gid] += scale * (dOut[gid] - x[(int)gid] * scale * dotDoutOut / (float)n);
}

// reluBackward: dIn[i] += dOut[i] if preRelu[i]>0 else 0
// Mirrors Value relu localGrad: [data > 0 ? 1.0 : 0.0]
kernel void reluBackward(
	device const float* preRelu [[ buffer(0) ]],
	device const float* dOut    [[ buffer(1) ]],
	device       float* dx      [[ buffer(2) ]],
	uint gid [[ thread_position_in_grid ]])
{
	dx[gid] += (preRelu[gid] > 0.0f) ? dOut[gid] : 0.0f;
}

// ─── UTILITY KERNELS ─────────────────────────────────────────────────────────

// addInPlace: dst[i] += src[i]  -- residual gradient accumulation
kernel void addInPlace(
	device       float* dst [[ buffer(0) ]],
	device const float* src [[ buffer(1) ]],
	uint gid [[ thread_position_in_grid ]])
{
	dst[gid] += src[gid];
}

// zeroBuffer: dst[i] = 0  -- zero gradient buffers between steps
kernel void zeroBuffer(
	device float* buf [[ buffer(0) ]],
	uint gid [[ thread_position_in_grid ]])
{
	buf[gid] = 0.0f;
}

// copyBuffer: dst[i] = src[i]
kernel void copyBuffer(
	device const float* src [[ buffer(0) ]],
	device       float* dst [[ buffer(1) ]],
	uint gid [[ thread_position_in_grid ]])
{
	dst[gid] = src[gid];
}

// ─── ADAM UPDATE ─────────────────────────────────────────────────────────────

// adamUpdate: mirrors Adam optimizer loop exactly.
// m[i] = beta1*m[i] + (1-beta1)*grad[i]
// v[i] = beta2*v[i] + (1-beta2)*grad[i]^2
// mHat = m[i] / (1-beta1^t),  vHat = v[i] / (1-beta2^t)
// param[i] -= lr * mHat / (sqrt(vHat) + eps)
kernel void adamUpdate(
	device       float* params [[ buffer(0) ]],
	device const float* grads  [[ buffer(1) ]],
	device       float* m      [[ buffer(2) ]],
	device       float* v      [[ buffer(3) ]],
	constant     float& lr     [[ buffer(4) ]],
	constant     float& beta1  [[ buffer(5) ]],
	constant     float& beta2  [[ buffer(6) ]],
	constant     float& eps    [[ buffer(7) ]],
	constant     float& beta1t [[ buffer(8) ]],   // beta1^step, precomputed on CPU
	constant     float& beta2t [[ buffer(9) ]],   // beta2^step, precomputed on CPU
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

// MARK: - [NEW - Metal] Metal Context

// MetalContext: holds all GPU state. Initialized once at program start.
// If Metal is unavailable, gpuAvailable=false and all GPU calls fall through to CPU.
class MetalContext {
	var gpuAvailable: Bool = false
	var device: MTLDevice?
	var commandQueue: MTLCommandQueue?
	var library: MTLLibrary?
	var pipelines: [String: MTLComputePipelineState] = [:]
	
	init() {
		guard let dev = MTLCreateSystemDefaultDevice() else {
			print("[Metal] No Metal device found — falling back to CPU"); return
		}
		guard let queue = dev.makeCommandQueue() else {
			print("[Metal] Could not create command queue — falling back to CPU"); return
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
				guard let fn = lib.makeFunction(name: name) else {
					print("[Metal] Missing kernel: \(name) — falling back to CPU"); return
				}
				pipelines[name] = try dev.makeComputePipelineState(function: fn)
			}
			self.device = dev; self.commandQueue = queue; self.library = lib
			self.gpuAvailable = true
			print("Metal: [GPU initialized: \(dev.name)]")
		} catch {
			print("[Metal] Shader compilation failed: \(error) — falling back to CPU")
		}
	}
	
	func makeBuffer(_ data: [Float]) -> MTLBuffer? {
		device?.makeBuffer(bytes: data, length: data.count * 4, options: .storageModeShared)
	}
	func makeBuffer(count: Int) -> MTLBuffer? {
		device?.makeBuffer(length: count * 4, options: .storageModeShared)
	}
	func readBuffer(_ buf: MTLBuffer, count: Int) -> [Float] {
		let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
		return Array(UnsafeBufferPointer(start: ptr, count: count))
	}
	func scalarInt(_ val: Int) -> MTLBuffer? {
		var v = Int32(val); return device?.makeBuffer(bytes: &v, length: 4, options: .storageModeShared)
	}
	func scalarFloat(_ val: Float) -> MTLBuffer? {
		var v = val; return device?.makeBuffer(bytes: &v, length: 4, options: .storageModeShared)
	}
	
	// Dispatch a kernel synchronously — result guaranteed ready when function returns
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
		cmdBuf.commit()
		cmdBuf.waitUntilCompleted()
	}
}

// Global Metal context — initialized once
var metal = MetalContext()

// MARK: - [NEW - Metal] GPU Forward Wrappers
// Used in: validation mode (compare against CPU) and generation mode (no backward needed).
// NOT used in GPU training — GPUTrainer handles that end-to-end with explicit grads.

func linearGPU(_ x: [Value], _ w: [[Value]]) -> [Value] {
	guard metal.gpuAvailable else { return linear(x, w) }
	let nin = x.count; let nout = w.count
	let xF = x.map { Float($0.data) }
	let wF = w.flatMap { $0.map { Float($0.data) } }
	guard let xB = metal.makeBuffer(xF), let wB = metal.makeBuffer(wF),
		  let oB = metal.makeBuffer(count: nout),
		  let ninB = metal.scalarInt(nin), let noutB = metal.scalarInt(nout)
	else { return linear(x, w) }
	metal.dispatch(kernel: "matVecMul", buffers: [xB, wB, oB, ninB, noutB], threadCount: nout)
	return metal.readBuffer(oB, count: nout).map { Value(Double($0)) }
}

func softmaxGPU(_ logits: [Value]) -> [Value] {
	guard metal.gpuAvailable else { return softmax(logits) }
	let n = logits.count
	let lF = logits.map { Float($0.data) }
	guard let lB = metal.makeBuffer(lF), let pB = metal.makeBuffer(count: n),
		  let nB = metal.scalarInt(n) else { return softmax(logits) }
	metal.dispatch(kernel: "softmaxForward", buffers: [lB, pB, nB], threadCount: n)
	return metal.readBuffer(pB, count: n).map { Value(Double($0)) }
}

func rmsnormGPU(_ x: [Value]) -> [Value] {
	guard metal.gpuAvailable else { return rmsnorm(x) }
	let n = x.count
	let xF = x.map { Float($0.data) }
	guard let xB = metal.makeBuffer(xF), let oB = metal.makeBuffer(count: n),
		  let nB = metal.scalarInt(n) else { return rmsnorm(x) }
	metal.dispatch(kernel: "rmsNormForward", buffers: [xB, oB, nB], threadCount: n)
	return metal.readBuffer(oB, count: n).map { Value(Double($0)) }
}

func reluGPU(_ x: [Value]) -> [Value] {
	guard metal.gpuAvailable else { return x.map { $0.relu() } }
	let n = x.count
	let xF = x.map { Float($0.data) }
	guard let xB = metal.makeBuffer(xF), let oB = metal.makeBuffer(count: n)
	else { return x.map { $0.relu() } }
	metal.dispatch(kernel: "reluForward", buffers: [xB, oB], threadCount: n)
	return metal.readBuffer(oB, count: n).map { Value(Double($0)) }
}

// gptGPUValidate: GPU forward pass using Value wrappers — used in --validate mode only.
// Returns logits as [Value] for loss comparison. Does NOT build a valid autograd graph.
func gptGPUValidate(tokenId: Int, posId: Int,
					keys: inout [[[Value]]], values: inout [[[Value]]],
					nEmbd: Int, nHead: Int, nLayer: Int) -> [Value] {
	let headDim = nEmbd / nHead
	let tokEmb = stateDict["wte"]![tokenId]
	let posEmb = stateDict["wpe"]![posId]
	var x = zip(tokEmb, posEmb).map(+)
	x = rmsnormGPU(x)
	for li in 0..<nLayer {
		let xResidual = x
		x = rmsnormGPU(x)
		let q = linearGPU(x, stateDict["layer\(li).attn_wq"]!)
		let k = linearGPU(x, stateDict["layer\(li).attn_wk"]!)
		let v = linearGPU(x, stateDict["layer\(li).attn_wv"]!)
		keys[li].append(k); values[li].append(v)
		var xAttn: [Value] = []
		for h in 0..<nHead {
			let hs = h * headDim
			let qH = Array(q[hs..<hs+headDim])
			let kH = keys[li].map { Array($0[hs..<hs+headDim]) }
			let vH = values[li].map { Array($0[hs..<hs+headDim]) }
			let attnLogits = kH.map { kt in
				zip(qH, kt).map(*).reduce(Value(0), +) / sqrt(Double(headDim))
			}
			let attnWeights = softmaxGPU(attnLogits)
			let headOut = (0..<headDim).map { j in
				zip(attnWeights, vH).map { w, vt in w * vt[j] }.reduce(Value(0), +)
			}
			xAttn.append(contentsOf: headOut)
		}
		x = linearGPU(xAttn, stateDict["layer\(li).attn_wo"]!)
		x = zip(x, xResidual).map(+)
		let xResidual2 = x
		x = rmsnormGPU(x)
		x = linearGPU(x, stateDict["layer\(li).mlp_fc1"]!)
		x = reluGPU(x)
		x = linearGPU(x, stateDict["layer\(li).mlp_fc2"]!)
		x = zip(x, xResidual2).map(+)
	}
	return linearGPU(x, stateDict["lm_head"]!)
}

// MARK: - [NEW - Metal] GPUTrainer — Milestone 2: Full GPU Forward + Backward + Adam

// GPUTrainer manages all persistent GPU buffers for one training run.
// Weight buffers, gradient buffers, moment buffers, and per-position activation
// buffers are all MTLBuffers in unified memory (M1: CPU and GPU share same RAM).
//
// Design: Option A (per-position sequential).
// We loop over sequence positions on CPU, dispatching Metal kernels for each
// position's forward and backward. Gradients accumulate in persistent dW buffers
// across positions, then Adam runs once per step on the full accumulated gradient.
// This matches the structure of the original training loop exactly.
class GPUTrainer {
	
	// ── Model dimensions ─────────────────────────────────────────────────────
	let E: Int        // nEmbd
	let H: Int        // nHead
	let L: Int        // nLayer
	let BS: Int       // blockSize
	let V: Int        // vocabSize
	let HD: Int       // headDim = E/H
	
	// ── Weight buffers (Float32) ─────────────────────────────────────────────
	var wte: MTLBuffer!      // [V * E]
	var wpe: MTLBuffer!      // [BS * E]
	var lmHead: MTLBuffer!   // [V * E]
	// Per layer: [attn_wq(E*E), attn_wk(E*E), attn_wv(E*E), attn_wo(E*E), mlp_fc1(4E*E), mlp_fc2(E*4E)]
	var lW: [[MTLBuffer]] = []
	
	// ── Gradient buffers (same shapes, zeroed each step) ────────────────────
	var dWte: MTLBuffer!
	var dWpe: MTLBuffer!
	var dLmHead: MTLBuffer!
	var lG: [[MTLBuffer]] = []
	
	// ── Adam moment buffers (persistent across steps, Float32) ───────────────
	var mWte: MTLBuffer!;    var vWte: MTLBuffer!
	var mWpe: MTLBuffer!;    var vWpe: MTLBuffer!
	var mLmHead: MTLBuffer!; var vLmHead: MTLBuffer!
	var lM: [[MTLBuffer]] = []
	var lV: [[MTLBuffer]] = []
	
	// Layer weight sizes: [E*E, E*E, E*E, E*E, 4E*E, E*4E]
	var lSizes: [Int] = []
	
	// ── Pre-allocated constant scalar buffers ────────────────────────────────
	// Every Int and Float constant the trainer ever passes to a kernel is
	// allocated ONCE at init time and reused forever. This eliminates the
	// thousands of ephemeral 4-byte MTLBuffer allocations that previously
	// exhausted the Metal heap after ~3000 steps.
	var cE: MTLBuffer!          // Int32(E)       — nEmbd
	var c4E: MTLBuffer!         // Int32(4*E)     — MLP hidden dim
	var cV: MTLBuffer!          // Int32(V)       — vocab size
	var cEE: MTLBuffer!         // Int32(E*E)     — used as nout for E*E mats (nout context)
	// We store one Int buffer and one Float buffer as mutable, rewritten each Adam step
	var cScaleF: MTLBuffer!     // Float — rewritten per position (1/seqLen)
	var cTargetI: MTLBuffer!    // Int32 — rewritten per position (target token id)
	// Adam scalar buffers — rewritten once per step
	var cLr: MTLBuffer!;   var cB1: MTLBuffer!;  var cB2: MTLBuffer!
	var cEps: MTLBuffer!;  var cB1t: MTLBuffer!; var cB2t: MTLBuffer!
	
	// ── Pre-allocated scratch working buffers ────────────────────────────────
	// All temporary buffers used inside forward() and backward() are pre-allocated
	// here and reused every step. No MTLBuffer is created during training.
	// Named by their role: sX = scratch buffer of size X floats.
	var sE1: MTLBuffer!; var sE2: MTLBuffer!; var sE3: MTLBuffer!   // size E — general scratch
	var sE4: MTLBuffer!; var sE5: MTLBuffer!; var sE6: MTLBuffer!
	var sE7: MTLBuffer!; var sE8: MTLBuffer!; var sE9: MTLBuffer!
	var sE10: MTLBuffer!; var sE11: MTLBuffer!; var sE12: MTLBuffer!
	var s4E1: MTLBuffer!; var s4E2: MTLBuffer!; var s4E3: MTLBuffer! // size 4E — MLP scratch
	var sV1: MTLBuffer!                                               // size V  — logits/probs

	init(nEmbd: Int, nHead: Int, nLayer: Int, blockSize: Int, vocabSize: Int) {
		E = nEmbd; H = nHead; L = nLayer; BS = blockSize; V = vocabSize; HD = nEmbd / nHead
		lSizes = [E*E, E*E, E*E, E*E, 4*E*E, E*4*E]
		setupBuffers()
		setupActBuffers()
	}
	
	// Allocate a permanent buffer of n floats (used at init only)
	func newBuf(_ n: Int) -> MTLBuffer {
		guard let b = metal.makeBuffer(count: n) else {
			fatalError("GPUTrainer: out of Metal memory allocating \(n*4) bytes at init")
		}
		return b
	}
	
	// Write an Int32 constant into a pre-allocated 4-byte buffer
	func setInt(_ buf: MTLBuffer, _ val: Int) {
		buf.contents().storeBytes(of: Int32(val), as: Int32.self)
	}
	// Write a Float32 constant into a pre-allocated 4-byte buffer
	func setFloat(_ buf: MTLBuffer, _ val: Float) {
		buf.contents().storeBytes(of: val, as: Float.self)
	}
	
	func setupBuffers() {
		// Persistent weight, gradient, and moment buffers
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
		
		// Pre-allocated constant scalar buffers (4 bytes each, written once or per-step)
		cE   = newBuf(1); setInt(cE,   E)
		c4E  = newBuf(1); setInt(c4E, 4*E)
		cV   = newBuf(1); setInt(cV,   V)
		cEE  = newBuf(1); setInt(cEE,  E)   // reused where nout=E context (nin arg)
		cScaleF  = newBuf(1)   // written per position
		cTargetI = newBuf(1)   // written per position
		cLr  = newBuf(1); cB1  = newBuf(1); cB2  = newBuf(1)
		cEps = newBuf(1); cB1t = newBuf(1); cB2t = newBuf(1)
		
		// Pre-allocated scratch working buffers — reused every forward/backward call
		// All temporary intermediate activations and gradients go here.
		// Size E: token/position embeddings, hidden states, attention outputs, grad vectors
		sE1 = newBuf(E); sE2 = newBuf(E); sE3 = newBuf(E)
		sE4 = newBuf(E); sE5 = newBuf(E); sE6 = newBuf(E)
		sE7 = newBuf(E); sE8 = newBuf(E); sE9 = newBuf(E)
		sE10 = newBuf(E); sE11 = newBuf(E); sE12 = newBuf(E)
		// Size 4E: MLP hidden layer (fc1 output before relu)
		s4E1 = newBuf(4*E); s4E2 = newBuf(4*E); s4E3 = newBuf(4*E)
		// Size V: logits and probabilities
		sV1 = newBuf(V)
	}
	
	// Sync Value.data (Double) -> MTLBuffer (Float32) before each step
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
	
	// Sync MTLBuffer (Float32) -> Value.data (Double) after Adam, zero grads
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
	
	// Zero all gradient buffers at the start of each training step
	func zeroGrads() {
		let all: [MTLBuffer] = [dWte, dWpe, dLmHead] + lG.flatMap { $0 }
		for b in all { metal.dispatch(kernel: "zeroBuffer", buffers: [b], threadCount: b.length/4) }
	}
	
	// ── Primitive GPU ops ────────────────────────────────────────────────────
	// All kernel dispatches use pre-allocated constant buffers — zero heap allocation.
	
	func linFwd(_ x: MTLBuffer, _ w: MTLBuffer, nin: Int, nout: Int, out: MTLBuffer) {
		let ninB:  MTLBuffer = (nin  == E) ? cE : c4E
		let noutB: MTLBuffer = (nout == E) ? cE : (nout == 4*E ? c4E : cV)
		metal.dispatch(kernel: "matVecMul",
					   buffers: [x, w, out, ninB, noutB],
					   threadCount: nout)
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
	
	// Linear backward: accumulates dW in-place, writes dx into provided buffer
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
	
	// Copy src -> dst (both same size n)
	func copyBuf(_ src: MTLBuffer, _ dst: MTLBuffer, n: Int) {
		metal.dispatch(kernel: "copyBuffer", buffers: [src, dst], threadCount: n)
	}
	
	// Zero a buffer
	func zeroBuf(_ buf: MTLBuffer, n: Int) {
		metal.dispatch(kernel: "zeroBuffer", buffers: [buf], threadCount: n)
	}
	
	// Embedding lookup: copy row from table into new buffer (CPU pointer, cheap for E=64)
	// Embedding lookup: copy row from table into a provided destination buffer (no allocation)
	func embLookup(_ table: MTLBuffer, row: Int, dim: Int, into dst: MTLBuffer) {
		dst.contents().copyMemory(from: table.contents().advanced(by: row * dim * 4), byteCount: dim * 4)
	}
	
	// Embedding gradient: dTable[row] += dVec (CPU pointer accumulation, cheap for E=64)
	func embBwd(_ dTable: MTLBuffer, _ dVec: MTLBuffer, row: Int, dim: Int) {
		let src = dVec.contents().bindMemory(to: Float.self, capacity: dim)
		let dst = dTable.contents().bindMemory(to: Float.self, capacity: dTable.length/4)
		for j in 0..<dim { dst[row * dim + j] += src[j] }
	}
	
	// ── Saved activations per layer per position ─────────────────────────────
	// MTLBuffer references point into pre-allocated layerActBufs — no heap allocation.
	struct LayerActs {
		var xIn: MTLBuffer         // input to this layer (before attn rmsnorm)
		var xNorm1: MTLBuffer      // after attn rmsnorm
		var q: MTLBuffer           // query projection output
		var attnW: [[Float]]       // softmax attention weights per head (CPU floats)
		var xAttn: MTLBuffer       // weighted value sum (concatenated heads)
		var xNorm2: MTLBuffer      // after MLP rmsnorm
		var preRelu: MTLBuffer     // MLP fc1 output before relu
		var xAfterAttn: MTLBuffer  // after attn residual (input to MLP block)
	}
	
	// Per-layer persistent activation buffers — allocated once at init, reused every step.
	// Each layer: 10 size-E slots + 1 size-4E slot (preRelu)
	// Shared forward state: fwdX (current x), fwdXPre0 (pre-norm0 embedding sum), fwdLogits
	var layerActBufs: [[MTLBuffer]] = []   // [layer][0..9], size E
	var layerMlpBufs: [[MTLBuffer]] = []   // [layer][0],    size 4E (preRelu)
	var fwdX:      MTLBuffer!   // current x flowing through layers (size E)
	var fwdXPre0:  MTLBuffer!   // pre-first-rmsnorm embedding sum (size E)
	var fwdLogits: MTLBuffer!   // lm_head output (size V)
	
	func setupActBuffers() {
		// Slot map per layer: [xIn, xNorm1, q, k, v, xAttn, xAttnProj, xAfterAttn, xNorm2, xMlp]
		for _ in 0..<L {
			layerActBufs.append((0..<10).map { _ in newBuf(E) })
			layerMlpBufs.append([newBuf(4*E)])   // preRelu
		}
		fwdX      = newBuf(E)
		fwdXPre0  = newBuf(E)
		fwdLogits = newBuf(V)
	}
	
	// ── Full GPU forward for one token position ──────────────────────────────
	// Zero heap allocation: all intermediate values go into pre-allocated buffers.
	// acts[] is populated with buffer references into layerActBufs for backward.
	func forward(tokenId: Int, posId: Int,
				 kCache: inout [[MTLBuffer]], vCache: inout [[MTLBuffer]],
				 acts: inout [LayerActs]) {
		
		// Token + position embeddings -> xPre0, then rmsnorm -> fwdX
		embLookup(wte, row: tokenId, dim: E, into: sE1)   // sE1 = tokEmb
		embLookup(wpe, row: posId,   dim: E, into: sE2)   // sE2 = posEmb
		copyBuf(sE1, fwdXPre0, n: E)
		addIP(fwdXPre0, sE2, n: E)                         // fwdXPre0 = tokEmb + posEmb
		rnFwd(fwdXPre0, n: E, out: fwdX)                  // fwdX = rmsnorm(xPre0)
		
		for li in 0..<L {
			let ab = layerActBufs[li]   // [xIn,xNorm1,q,k,v,xAttn,xAttnProj,xAfterAttn,xNorm2,xMlp]
			let mb = layerMlpBufs[li]   // [preRelu]
			
			copyBuf(fwdX, ab[0], n: E)                            // ab[0] = xIn (save for residual)
			rnFwd(fwdX, n: E, out: ab[1])                         // ab[1] = xNorm1
			linFwd(ab[1], lW[li][0], nin: E, nout: E, out: ab[2]) // ab[2] = q
			linFwd(ab[1], lW[li][1], nin: E, nout: E, out: ab[3]) // ab[3] = k
			linFwd(ab[1], lW[li][2], nin: E, nout: E, out: ab[4]) // ab[4] = v
			kCache[li].append(ab[3])
			vCache[li].append(ab[4])
			
			// Multi-head attention (CPU-side: small per-head vectors, no GPU kernel needed)
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
			ab[5].contents().copyMemory(from: xAttnF, byteCount: E * 4) // ab[5] = xAttn
			
			linFwd(ab[5], lW[li][3], nin: E, nout: E, out: ab[6]) // ab[6] = xAttnProj
			copyBuf(ab[6], ab[7], n: E)
			addIP(ab[7], ab[0], n: E)                              // ab[7] = xAfterAttn (attn residual)
			
			rnFwd(ab[7], n: E, out: ab[8])                             // ab[8] = xNorm2
			linFwd(ab[8], lW[li][4], nin: E, nout: 4*E, out: mb[0])   // mb[0] = preRelu
			reluFwd(mb[0], n: 4*E, out: s4E1)                         // s4E1  = xRelu (temp)
			linFwd(s4E1, lW[li][5], nin: 4*E, nout: E, out: ab[9])    // ab[9] = xMlp
			copyBuf(ab[9], fwdX, n: E)
			addIP(fwdX, ab[7], n: E)                                   // fwdX = xMlp + xAfterAttn
			
			acts.append(LayerActs(xIn: ab[0], xNorm1: ab[1], q: ab[2],
								  attnW: attnWeightsAllHeads,
								  xAttn: ab[5], xNorm2: ab[8],
								  preRelu: mb[0], xAfterAttn: ab[7]))
		}
		linFwd(fwdX, lmHead, nin: E, nout: V, out: fwdLogits)   // fwdLogits = lm_head(x)
	}
	
	// ── Full GPU backward for one token position ─────────────────────────────
	// Reads dLogits from sV1 (written by softmaxCEBackward before this call).
	// Uses scratch buffers sE1..sE9, s4E1..s4E3 — zero heap allocation.
	func backward(posId: Int, tokenId: Int,
				  acts: [LayerActs],
				  kCache: [[MTLBuffer]], vCache: [[MTLBuffer]]) {
		
		// LM head backward: dLmHead += outer(dLogits, xFinal), dx -> sE1
		zeroBuf(sE1, n: E)
		linBwd(dOut: sV1, x: fwdX, w: lmHead, dW: dLmHead, nin: E, nout: V, dx: sE1)
		
		for li in stride(from: L-1, through: 0, by: -1) {
			let a = acts[li]
			
			// MLP residual split: sE1=dxMlp (direct), sE2=dxAfterAttn (copy — independent)
			copyBuf(sE1, sE2, n: E)
			
			// MLP backward (mlp_fc2 -> relu -> mlp_fc1)
			reluFwd(a.preRelu, n: 4*E, out: s4E1)                              // s4E1 = xRelu
			zeroBuf(s4E2, n: 4*E)
			linBwd(dOut: sE1, x: s4E1, w: lW[li][5], dW: lG[li][5], nin: 4*E, nout: E, dx: s4E2)
			zeroBuf(s4E3, n: 4*E)
			reluBwd(pre: a.preRelu, dOut: s4E2, n: 4*E, dx: s4E3)              // s4E3 = dxPreRelu
			zeroBuf(sE3, n: E)
			linBwd(dOut: s4E3, x: a.xNorm2, w: lW[li][4], dW: lG[li][4], nin: E, nout: 4*E, dx: sE3)
			// RMSNorm2 backward
			zeroBuf(sE4, n: E)
			rnBwd(x: a.xAfterAttn, dOut: sE3, n: E, dx: sE4)
			addIP(sE2, sE4, n: E)         // sE2 = full dxAfterAttn
			copyBuf(sE2, sE1, n: E)       // sE1 = dx flowing up
			
			// Attention residual split: sE1=dxAttnProj (direct), sE2=dxIn (copy — independent)
			copyBuf(sE1, sE2, n: E)
			
			// Attention projection backward (attn_wo)
			zeroBuf(sE3, n: E)
			linBwd(dOut: sE1, x: a.xAttn, w: lW[li][3], dW: lG[li][3], nin: E, nout: E, dx: sE3)
			
			// Attention backward (CPU-side: small per-head vectors)
			let dxAttnF = metal.readBuffer(sE3, count: E)
			var dqF = [Float](repeating: 0, count: E)
			var dkF = [Float](repeating: 0, count: E)
			var dvF = [Float](repeating: 0, count: E)
			for h in 0..<H {
				let hs = h * HD
				let aw = a.attnW[h]; let seqLen = aw.count
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
			// Write dQ/dK/dV into scratch buffers, then backward through Q/K/V projections
			sE4.contents().copyMemory(from: dqF, byteCount: E * 4)
			sE5.contents().copyMemory(from: dkF, byteCount: E * 4)
			sE6.contents().copyMemory(from: dvF, byteCount: E * 4)
			zeroBuf(sE7, n: E); zeroBuf(sE8, n: E); zeroBuf(sE9, n: E)
			linBwd(dOut: sE4, x: a.xNorm1, w: lW[li][0], dW: lG[li][0], nin: E, nout: E, dx: sE7)
			linBwd(dOut: sE5, x: a.xNorm1, w: lW[li][1], dW: lG[li][1], nin: E, nout: E, dx: sE8)
			linBwd(dOut: sE6, x: a.xNorm1, w: lW[li][2], dW: lG[li][2], nin: E, nout: E, dx: sE9)
			addIP(sE7, sE8, n: E); addIP(sE7, sE9, n: E)   // sE7 = dxNorm1 total
			// RMSNorm1 backward
			zeroBuf(sE8, n: E)
			rnBwd(x: a.xIn, dOut: sE7, n: E, dx: sE8)
			addIP(sE2, sE8, n: E)         // sE2 = full dxIn
			copyBuf(sE2, sE1, n: E)       // sE1 = dx for next layer
		}
		
		// Embedding backward: backprop through first rmsnorm
		zeroBuf(sE2, n: E)
		rnBwd(x: fwdXPre0, dOut: sE1, n: E, dx: sE2)
		embBwd(dWte, sE2, row: tokenId, dim: E)
		embBwd(dWpe, sE2, row: posId,   dim: E)
	}

	// ── GPU Adam update for all parameters ───────────────────────────────────
	// Uses pre-allocated cLr/cB1/cB2/cEps/cB1t/cB2t — zero heap allocation per step.
	func adamStep(step: Int, lr: Float, beta1: Float, beta2: Float, eps: Float) {
		let b1t = Float(Foundation.pow(Double(beta1), Double(step)))
		let b2t = Float(Foundation.pow(Double(beta2), Double(step)))
		// Write current step's values into pre-allocated scalar buffers
		setFloat(cLr,  lr);   setFloat(cB1,  beta1); setFloat(cB2,  beta2)
		setFloat(cEps, eps);  setFloat(cB1t, b1t);   setFloat(cB2t, b2t)
		let sc: [MTLBuffer] = [cLr, cB1, cB2, cEps, cB1t, cB2t]
		func upd(_ p: MTLBuffer, _ g: MTLBuffer, _ m: MTLBuffer, _ v: MTLBuffer, _ n: Int) {
			metal.dispatch(kernel: "adamUpdate", buffers: [p, g, m, v] + sc, threadCount: n)
		}
		upd(wte, dWte, mWte, vWte, V*E)
		upd(wpe, dWpe, mWpe, vWpe, BS*E)
		upd(lmHead, dLmHead, mLmHead, vLmHead, V*E)
		for li in 0..<L {
			for wi in 0..<6 { upd(lW[li][wi], lG[li][wi], lM[li][wi], lV[li][wi], lSizes[wi]) }
		}
	}
}

// MARK: - [ORIGINAL - untouched] Model Architecture

var stateDict: [String: [[Value]]] = [:]

func gpt(tokenId: Int, posId: Int, keys: inout [[[Value]]], values: inout [[[Value]]],
		 nEmbd: Int, nHead: Int, nLayer: Int) -> [Value] {
	let headDim = nEmbd / nHead
	
	let tokEmb = stateDict["wte"]![tokenId]
	let posEmb = stateDict["wpe"]![posId]
	var x = zip(tokEmb, posEmb).map(+)
	x = rmsnorm(x)
	
	for li in 0..<nLayer {
		// Multi-head attention
		// NOTE: Each attention head is independent and could run in parallel,
		// but coordination overhead exceeds benefits for this model size.
		// For larger models (8+ heads), parallel dispatch would help.
		let xResidual = x
		x = rmsnorm(x)
		let q = linear(x, stateDict["layer\(li).attn_wq"]!)
		let k = linear(x, stateDict["layer\(li).attn_wk"]!)
		let v = linear(x, stateDict["layer\(li).attn_wv"]!)
		keys[li].append(k)
		values[li].append(v)
		
		var xAttn: [Value] = []
		for h in 0..<nHead {
			let hs = h * headDim
			let qH = Array(q[hs..<hs+headDim])
			let kH = keys[li].map { Array($0[hs..<hs+headDim]) }
			let vH = values[li].map { Array($0[hs..<hs+headDim]) }
			
			let attnLogits = kH.map { kt in
				zip(qH, kt).map(*).reduce(Value(0), +) / sqrt(Double(headDim))
			}
			let attnWeights = softmax(attnLogits)
			let headOut = (0..<headDim).map { j in
				zip(attnWeights, vH).map { w, vt in w * vt[j] }.reduce(Value(0), +)
			}
			xAttn.append(contentsOf: headOut)
		}
		
		x = linear(xAttn, stateDict["layer\(li).attn_wo"]!)
		x = zip(x, xResidual).map(+)
		
		// MLP block
		let xResidual2 = x
		x = rmsnorm(x)
		x = linear(x, stateDict["layer\(li).mlp_fc1"]!)
		x = x.map { $0.relu() }
		x = linear(x, stateDict["layer\(li).mlp_fc2"]!)
		x = zip(x, xResidual2).map(+)
	}
	
	let logits = linear(x, stateDict["lm_head"]!)
	return logits
}

// MARK: - [NEW - Metal] Validation and Timestamp Utilities

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

// MARK: - [ORIGINAL - untouched] Main Program

print("\n--| MicroGPT 👾 Swift Edition |-----")

let args = CommandLine.arguments
var mode = "train"
var inputFiles: [String] = []
var prompt = ""
var modelPath = "model.txt"
var numSteps = 1000
var temperature = 0.5
var maxSamples = 100000

// [NEW - Metal] Additional CLI flags
var useGPU = false        // --gpu: full GPU forward+backward+Adam (Milestone 2)
var validateMode = false  // --validate: run CPU and GPU forward in parallel, compare loss

var i = 1
while i < args.count {
	switch args[i] {
	case "--train":    mode = "train"
	case "--generate": mode = "generate"
	case "--prompt":   i += 1; if i < args.count { prompt = args[i] }
	case "--model":    i += 1; if i < args.count { modelPath = args[i] }
	case "--steps":    i += 1; if i < args.count { numSteps = Int(args[i]) ?? 1000 }
	case "--temperature", "--temp": i += 1; if i < args.count { temperature = Double(args[i]) ?? 0.5 }
	case "--samples":  i += 1; if i < args.count { maxSamples = Int(args[i]) ?? 100000 }
	case "--gpu":      useGPU = true
	case "--validate": validateMode = true
	default: if args[i].hasSuffix(".txt") { inputFiles.append(args[i]) }
	}
	i += 1
}

// Model hyperparameters
let nEmbd = 64      // Increased from 32 (embedding dimension)
let nHead = 4       // Same (attention heads)
let nLayer = 4      // Increased from 2 (transformer layers — 4x depth!)
let blockSize = 80  // Context window (can see 80 characters)

if mode == "train" {
	
	let trainStart = Date()
	print("Training started: \(formattedTime(trainStart))")
	
	var docs: [String] = []
	
	if inputFiles.isEmpty {
		print("Usage for training:")
		print("  swiftc -o microgpt3 microgpt3.swift -framework Metal -framework Foundation")
		print("  ./microgpt3 --train file.txt [--steps 1000] [--gpu] [--validate] [--model out.txt]")
		print("\nNo input files provided. Using example data...")
		docs = ["hello world", "swift is great", "machine learning"]
	} else {
		print("Loading files...")
		for file in inputFiles {
			let fileURL = URL(fileURLWithPath: (file as NSString).expandingTildeInPath)
			let filePath = fileURL.path
			let currentDirPath = FileManager.default.currentDirectoryPath + "/" + file
			var content: String? = nil
			if FileManager.default.fileExists(atPath: filePath) {
				content = try? String(contentsOfFile: filePath, encoding: .utf8)
			} else if FileManager.default.fileExists(atPath: currentDirPath) {
				content = try? String(contentsOfFile: currentDirPath, encoding: .utf8)
			} else if FileManager.default.fileExists(atPath: file) {
				content = try? String(contentsOfFile: file, encoding: .utf8)
			}
			if let content = content {
				let lines = content.components(separatedBy: .newlines)
					.map { $0.trimmingCharacters(in: .whitespaces) }
					.filter { !$0.isEmpty }
					.filter { $0.count <= 200 }
				docs.append(contentsOf: lines)
				print("  ✓ Loaded \(file): \(lines.count) lines")
			} else {
				print("  ✗ Could not read file: \(file)")
				print("    Current directory: \(FileManager.default.currentDirectoryPath)")
			}
		}
		if docs.isEmpty {
			print("\n❌ Error: No documents loaded!")
			print("Check file exists and is readable in current directory.")
			exit(1)
		}
		if docs.count > maxSamples {
			print("\nDataset is large (\(docs.count) lines). Sampling \(maxSamples) lines for training...")
			docs = Array(docs.shuffled().prefix(maxSamples))
		}
	}
	
	docs.shuffle()
	print("Loaded \(docs.count) documents")
	
	let allChars = Set(docs.joined())
	let uchars = allChars.sorted()
	let BOS = uchars.count
	let vocabSize = uchars.count + 1
	
	print("Vocabulary size: \(vocabSize)")
	print("Characters: \(String(uchars))")
	
	// [NEW - Metal] Report training mode clearly
	if useGPU && !metal.gpuAvailable {
		print("Training mode: CPU (Metal unavailable — --gpu flag ignored)")
		useGPU = false
	} else if useGPU {
		print("Training mode: GPU — full forward + backward + Adam on Metal (Milestone 2)")
	} else {
		print("Training mode: CPU (original Value{} autograd)")
	}
	if validateMode {
		print("Validation mode: ON — CPU loss vs GPU loss printed each step")
	}
	
	// Initialize weights (one set, shared between CPU and GPU paths)
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
	print("Number of parameters: \(params.count)\n")
	
	// Snapshot initial weights for validation (GPU path needs same starting point as CPU)
	var gpuSnapshot: [String: [[Double]]] = [:]
	if validateMode {
		for (key, mat) in stateDict { gpuSnapshot[key] = mat.map { $0.map { $0.data } } }
	}
	
	// CPU Adam buffers
	var adamM = [Double](repeating: 0.0, count: params.count)
	var adamV = [Double](repeating: 0.0, count: params.count)
	let learningRate = 0.01
	let beta1 = 0.85
	let beta2 = 0.99
	let epsAdam = 1e-8
	
	// GPU trainer (Milestone 2)
	var trainer: GPUTrainer? = nil
	if useGPU && metal.gpuAvailable {
		trainer = GPUTrainer(nEmbd: nEmbd, nHead: nHead, nLayer: nLayer,
							 blockSize: blockSize, vocabSize: vocabSize)
	}
	
	var valPasses = 0; var valFails = 0
	
	print("Training for \(numSteps) steps...")
	let loopStart = Date()
	
	for step in 0..<numSteps {
		let doc = docs[step % docs.count]
		var tokens = [BOS]
		tokens.append(contentsOf: doc.map { ch in uchars.firstIndex(of: ch)! })
		tokens.append(BOS)
		let seqLen = min(blockSize, tokens.count - 1)
		
		var stepLoss = 0.0
		
		if let tr = trainer, useGPU && !validateMode {
			// ── FULL GPU TRAINING PATH (Milestone 2) ─────────────────────────
			tr.syncFrom(stateDict)    // CPU weights -> GPU buffers
			tr.zeroGrads()            // zero all gradient accumulators
			
			var kCache = [[MTLBuffer]](repeating: [], count: nLayer)
			var vCache = [[MTLBuffer]](repeating: [], count: nLayer)
			var totalLoss: Float = 0
			
			for posId in 0..<seqLen {
				let tokenId  = tokens[posId]
				let targetId = tokens[posId + 1]
				
				var posActs: [GPUTrainer.LayerActs] = []
				tr.forward(tokenId: tokenId, posId: posId,
						   kCache: &kCache, vCache: &vCache, acts: &posActs)
				
				// Compute loss on CPU from fwdLogits (single scalar, negligible cost)
				let logitsF = metal.readBuffer(tr.fwdLogits, count: vocabSize)
				let maxL = logitsF.max()!
				var exps = logitsF.map { Foundation.exp(Double($0 - maxL)) }
				let sumE = exps.reduce(0, +); exps = exps.map { $0 / sumE }
				totalLoss += Float(-Foundation.log(max(exps[targetId], 1e-10)))
				
				// Softmax -> sV1, then dLogits -> sV1 via softmaxCEBackward
				// (sV1 is the pre-allocated vocabSize buffer backward() reads from)
				tr.smFwd(tr.fwdLogits, n: vocabSize, out: tr.sV1)
				// Write target and scale into pre-allocated mutable scalar buffers
				tr.setInt(tr.cTargetI, targetId)
				tr.setFloat(tr.cScaleF, 1.0 / Float(seqLen))
				metal.dispatch(kernel: "softmaxCEBackward",
							   buffers: [tr.sV1, tr.sV1, tr.cTargetI, tr.cV, tr.cScaleF],
							   threadCount: vocabSize)
				// Note: softmaxCEBackward reads probs[gid] and writes dLogits[gid] —
				// both point to sV1, which is safe because each thread reads then writes its own element.
				
				// Full GPU backward — reads dLogits from sV1
				tr.backward(posId: posId, tokenId: tokenId,
							acts: posActs, kCache: kCache, vCache: vCache)
			}
			
			stepLoss = Double(totalLoss / Float(seqLen))
			
			// GPU Adam update (one dispatch per parameter matrix)
			let lrT = Float(learningRate * (1.0 - Double(step) / Double(numSteps)))
			tr.adamStep(step: step + 1, lr: lrT,
						beta1: Float(beta1), beta2: Float(beta2), eps: Float(epsAdam))
			
			// GPU weights -> CPU Value objects (so model can be saved/used in generation)
			tr.syncTo(stateDict)
			
		} else {
			// ── CPU TRAINING PATH (original, authoritative) ───────────────────
			var keys   = [[[Value]]](repeating: [], count: nLayer)
			var values = [[[Value]]](repeating: [], count: nLayer)
			var losses: [Value] = []
			
			for posId in 0..<seqLen {
				let tokenId  = tokens[posId]
				let targetId = tokens[posId + 1]
				let logits = gpt(tokenId: tokenId, posId: posId,
								 keys: &keys, values: &values,
								 nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
				let probs = softmax(logits)
				losses.append(-probs[targetId].log())
			}
			
			let loss = losses.reduce(Value(0), +) / Double(seqLen)
			stepLoss = loss.data
			loss.backward()
			
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
		
		// ── VALIDATION: GPU forward vs CPU forward ────────────────────────────
		if validateMode {
			// Use snapshot weights (same as what CPU used this step, before update)
			var gpuSD: [String: [[Value]]] = [:]
			for (key, mat) in gpuSnapshot { gpuSD[key] = mat.map { $0.map { Value($0) } } }
			let saved = stateDict; stateDict = gpuSD
			
			var gpuKeys = [[[Value]]](repeating: [], count: nLayer)
			var gpuVals = [[[Value]]](repeating: [], count: nLayer)
			var gpuLosses: [Value] = []
			for posId in 0..<seqLen {
				let tokenId  = tokens[posId]; let targetId = tokens[posId + 1]
				let gl = gptGPUValidate(tokenId: tokenId, posId: posId,
										keys: &gpuKeys, values: &gpuVals,
										nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
				gpuLosses.append(-softmax(gl)[targetId].log())
			}
			let gpuLoss = (gpuLosses.reduce(Value(0), +) / Double(seqLen)).data
			stateDict = saved
			
			// Advance snapshot to post-update weights for next step
			for (key, mat) in stateDict { gpuSnapshot[key] = mat.map { $0.map { $0.data } } }
			
			let passed = validateLoss(step: step + 1, cpuLoss: stepLoss, gpuLoss: gpuLoss)
			if passed { valPasses += 1 } else { valFails += 1 }
		}
		
		// Progress reporting — every 10 steps or first step
		if (step + 1) % 10 == 0 || step == 0 {
			let now = Date()
			let elapsed = now.timeIntervalSince(loopStart)
			let sps = Double(step + 1) / elapsed
			let rem = Double(numSteps - step - 1) / sps
			let h = Int(rem)/3600; let m = (Int(rem)%3600)/60; let s = Int(rem)%60
			let eta = h > 0 ? "\(h)h\(m)m" : m > 0 ? "\(m)m\(s)s" : "\(s)s"
			print(String(format: "step %4d / %4d | loss %.4f | %.2f steps/s | ETA: %@",
						step + 1, numSteps, stepLoss, sps, eta))
		}
	}
	
	// Training complete — timestamp summary
	let trainEnd = Date()
	let totalElapsed = trainEnd.timeIntervalSince(trainStart)
	let avgSPS = Double(numSteps) / trainEnd.timeIntervalSince(loopStart)
	print("\n--- Training Complete ----------")
	print("  Started:   \(formattedTime(trainStart))")
	print("  Finished:  \(formattedTime(trainEnd))")
	print("  Elapsed:   \(formattedElapsed(totalElapsed))")
	print(String(format: "  Avg speed: %.3f steps/s", avgSPS))
	if validateMode {
		print("  Validation: \(valPasses) PASS / \(valFails) FAIL out of \(numSteps) steps")
		print(valFails == 0
			? "  Result: ✓ GPU matches CPU within tolerance (\(validationTolerance))"
			: "  Result: ✗ GPU diverged on \(valFails) steps — check precision")
	}
	
	// Save model
	print("  Saving model to \(modelPath)...")
	var modelData = "\(vocabSize)\n\(String(uchars))\n"
	for (key, mat) in stateDict.sorted(by: { $0.key < $1.key }) {
		modelData += "\(key)\n"
		for row in mat { modelData += row.map { String($0.data) }.joined(separator: ",") + "\n" }
	}
	try? modelData.write(toFile: modelPath, atomically: true, encoding: .utf8)
	print("  Model saved!")
	
} else if mode == "generate" {
	
	print("Loading model from \(modelPath)...")
	guard let modelContent = try? String(contentsOfFile: modelPath, encoding: .utf8) else {
		print("Error: Could not load model file. Train a model first!"); exit(1)
	}
	
	let lines = modelContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
	let vocabSize = Int(lines[0])!
	let uchars = Array(lines[1])
	let BOS = uchars.count
	print("Vocabulary size: \(vocabSize)")
	
	let matrixSizes: [String: (Int, Int)] = [
		"wte": (vocabSize, nEmbd), "wpe": (blockSize, nEmbd), "lm_head": (vocabSize, nEmbd),
		"layer0.attn_wq": (nEmbd, nEmbd), "layer0.attn_wk": (nEmbd, nEmbd),
		"layer0.attn_wv": (nEmbd, nEmbd), "layer0.attn_wo": (nEmbd, nEmbd),
		"layer0.mlp_fc1": (4*nEmbd, nEmbd), "layer0.mlp_fc2": (nEmbd, 4*nEmbd),
		"layer1.attn_wq": (nEmbd, nEmbd), "layer1.attn_wk": (nEmbd, nEmbd),
		"layer1.attn_wv": (nEmbd, nEmbd), "layer1.attn_wo": (nEmbd, nEmbd),
		"layer1.mlp_fc1": (4*nEmbd, nEmbd), "layer1.mlp_fc2": (nEmbd, 4*nEmbd),
		"layer2.attn_wq": (nEmbd, nEmbd), "layer2.attn_wk": (nEmbd, nEmbd),
		"layer2.attn_wv": (nEmbd, nEmbd), "layer2.attn_wo": (nEmbd, nEmbd),
		"layer2.mlp_fc1": (4*nEmbd, nEmbd), "layer2.mlp_fc2": (nEmbd, 4*nEmbd),
		"layer3.attn_wq": (nEmbd, nEmbd), "layer3.attn_wk": (nEmbd, nEmbd),
		"layer3.attn_wv": (nEmbd, nEmbd), "layer3.attn_wo": (nEmbd, nEmbd),
		"layer3.mlp_fc1": (4*nEmbd, nEmbd), "layer3.mlp_fc2": (nEmbd, 4*nEmbd)
	]
	
	var lineIdx = 2
	while lineIdx < lines.count {
		let key = lines[lineIdx]; lineIdx += 1
		guard let (nrows, _) = matrixSizes[key] else { print("Warning: Unknown key \(key)"); continue }
		var mat: [[Value]] = []
		for _ in 0..<nrows {
			if lineIdx >= lines.count { break }
			mat.append(lines[lineIdx].split(separator: ",").map { Value(Double($0)!) })
			lineIdx += 1
		}
		if mat.count == nrows { stateDict[key] = mat }
		else { print("Warning: Matrix \(key) wrong size: got \(mat.count), expected \(nrows)") }
	}
	print("Model loaded!\n")
	
	func generate(from promptText: String, numSamples: Int = 1) {
		for sampleIdx in 0..<numSamples {
			var keys   = [[[Value]]](repeating: [], count: nLayer)
			var values = [[[Value]]](repeating: [], count: nLayer)
			var generated: [Character] = []
			var position = 0
			for ch in promptText {
				if let tokenId = uchars.firstIndex(of: ch), position < blockSize {
					_ = gpt(tokenId: tokenId, posId: position, keys: &keys, values: &values,
						   nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
					position += 1
				}
			}
			var tokenId = BOS
			for posId in position..<blockSize {
				let logits = gpt(tokenId: tokenId, posId: posId, keys: &keys, values: &values,
							   nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
				let probs = softmax(logits.map { $0 / temperature })
				let weights = probs.map { $0.data }
				let total = weights.reduce(0, +)
				var rand = Double.random(in: 0..<total)
				tokenId = 0
				for (idx, w) in weights.enumerated() { rand -= w; if rand <= 0 { tokenId = idx; break } }
				if tokenId == BOS { break }
				generated.append(uchars[tokenId])
			}
			if numSamples > 1 {
				FancyPrint.out(String(format: "  %d: %@%@", sampleIdx + 1, promptText, String(generated)))
			} else {
				FancyPrint.out("\(promptText)\(String(generated))")
			}
		}
	}
	
	print("═══════════════════════════════════════")
	print("  Interactive Generation Mode")
	print("═══════════════════════════════════════")
	print("Temperature: \(temperature)")
	print("Type your prompt and press Enter.")
	print("Commands:")
	print("  'exit' or 'quit' - Exit the program")
	print("  'multi <n>' - Generate n variations of next prompt")
	print("  'temp <n>' - Change temperature (0.1-1.0)")
	print("═══════════════════════════════════════\n")
	
	var numSamples = 1
	while true {
		print("> ", terminator: ""); fflush(stdout)
		guard let input = readLine() else { break }
		let trimmed = input.trimmingCharacters(in: .whitespaces)
		if trimmed.isEmpty { continue }
		if trimmed.lowercased() == "exit" || trimmed.lowercased() == "quit" { FancyPrint.out("\nGoodbye! 👋"); break }
		if trimmed.lowercased().hasPrefix("multi ") {
			if let n = Int(trimmed.dropFirst(6).trimmingCharacters(in: .whitespaces)) {
				numSamples = max(1, min(n, 20)); print("✓ Will generate \(numSamples) variations\n")
			} else { print("Usage: multi <number>\n") }
			continue
		}
		if trimmed.lowercased().hasPrefix("temp ") {
			if let t = Double(trimmed.dropFirst(5).trimmingCharacters(in: .whitespaces)) {
				temperature = max(0.1, min(t, 2.0)); print("✓ Temperature set to \(temperature)\n")
			} else { print("Usage: temp <number> (e.g., temp 0.7)\n") }
			continue
		}
		generate(from: trimmed, numSamples: numSamples); print("")
		if numSamples > 1 { numSamples = 1 }
	}
}

// Usage only printed when --help is requested or no arguments given
if args.contains("--help") || args.contains("-h") || args.count == 1 {
	print("Usage:")
	print("  ./microgpt3 --train file.txt [--steps 1000] [--model model.txt]")
	print("  ./microgpt3 --train file.txt --gpu   [--steps 1000] [--model model.txt]")
	print("  ./microgpt3 --train file.txt --validate [--steps 10]")
	print("  ./microgpt3 --generate [--temp 0.5] [--model model.txt]")
	print("")
	print("Options:")
	print("  --steps N     Number of training steps (default: 1000)")
	print("  --samples N   Max lines to load (default: 100000)")
	print("  --temp N      Temperature 0.1-2.0 for generation (default: 0.5)")
	print("  --model path  Model file path (default: model.txt)")
	print("  --gpu         Full GPU training: forward + backward + Adam on Metal")
	print("  --validate    Compare CPU vs GPU loss each step (runs CPU path)")
	print("  --help        Show this help message")
}
