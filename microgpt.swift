#!/usr/bin/env swift
// MicroGPT in Swift v2.0
// Ported by John Roland Penner
// February 15, 2026
/**
 The most atomic way to train and inference a GPT in pure Swift.
 This file is the complete algorithm.
 Everything else is just efficiency.
 
 Based on @karpathy's micrograd GPT
 Multi-threaded version for performance
 */

import Foundation

// MARK: - Autograd Value class

// Let there be Autograd, to recursively apply the chain rule through a computation graph
class Value {
	var data: Double
	var grad: Double = 0.0
	private var children: [Value] = []
	private var localGrads: [Double] = []
	
	init(_ data: Double, children: [Value] = [], localGrads: [Double] = []) {
		self.data = data		// scalar value of this node calculated during forward pass
		self.children = children		// derivative of the loss w.r.t. this node, calculated in backward pass
		self.localGrads = localGrads	// local derivative of this node w.r.t. its children
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

// MARK: - Helper Functions

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

// MARK: - Model Architecture

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

// MARK: - Main Program

print("MicroGPT - Swift Edition")
print("========================\n")

// Parse command line arguments
let args = CommandLine.arguments
var mode = "train"
var inputFiles: [String] = []
var prompt = ""
var modelPath = "model.txt"
var numSteps = 1000
var temperature = 0.5
var maxSamples = 100000 // Increased default from 50k to 100k

var i = 1
while i < args.count {
	switch args[i] {
	case "--train":
		mode = "train"
	case "--generate":
		mode = "generate"
	case "--prompt":
		i += 1
		if i < args.count {
			prompt = args[i]
		}
	case "--model":
		i += 1
		if i < args.count {
			modelPath = args[i]
		}
	case "--steps":
		i += 1
		if i < args.count {
			numSteps = Int(args[i]) ?? 1000
		}
	case "--temperature", "--temp":
		i += 1
		if i < args.count {
			temperature = Double(args[i]) ?? 0.5
		}
	case "--samples":
		i += 1
		if i < args.count {
			maxSamples = Int(args[i]) ?? 100000
		}
	default:
		if args[i].hasSuffix(".txt") {
			inputFiles.append(args[i])
		}
	}
	i += 1
}

// Model hyperparameters
let nEmbd = 64      // Increased from 32 (embedding dimension)
let nHead = 4       // Same (attention heads)
let nLayer = 4      // Increased from 2 (transformer layers - 4x depth!)
let blockSize = 80  // Context window (can see 80 characters)

if mode == "train" {
	// Load training data
	var docs: [String] = []
	
	if inputFiles.isEmpty {
		print("Usage for training:")
		print("  swiftc -o microgpt microgpt.swift")
		print("  ./microgpt --train file1.txt file2.txt file3.txt [--steps 1000] [--model model.txt]")
		print("\nNo input files provided. Using example data...")
		docs = ["hello world", "swift is great", "machine learning"]
	} else {
		print("Loading files...")
		for file in inputFiles {
			// Try to expand the file path
			let fileURL = URL(fileURLWithPath: (file as NSString).expandingTildeInPath)
			let filePath = fileURL.path
			
			// Also try current directory
			let currentDirPath = FileManager.default.currentDirectoryPath + "/" + file
			
			var content: String? = nil
			
			// Try original path
			if FileManager.default.fileExists(atPath: filePath) {
				content = try? String(contentsOfFile: filePath, encoding: .utf8)
			}
			// Try current directory
			else if FileManager.default.fileExists(atPath: currentDirPath) {
				content = try? String(contentsOfFile: currentDirPath, encoding: .utf8)
			}
			// Try just the filename as-is
			else if FileManager.default.fileExists(atPath: file) {
				content = try? String(contentsOfFile: file, encoding: .utf8)
			}
			
			if let content = content {
				let lines = content.components(separatedBy: .newlines)
					.map { $0.trimmingCharacters(in: .whitespaces) }
					.filter { !$0.isEmpty }
					.filter { $0.count <= 200 } // Skip very long lines to avoid memory issues
				docs.append(contentsOf: lines)
				print("  ✓ Loaded \(file): \(lines.count) lines")
			} else {
				print("  ✗ Could not read file: \(file)")
				print("    Looked in: \(filePath)")
				print("    Current directory: \(FileManager.default.currentDirectoryPath)")
			}
		}
		
		// Check if we got any data
		if docs.isEmpty {
			print("\n❌ Error: No documents loaded! Cannot train on empty dataset.")
			print("Please check:")
			print("  1. File exists in current directory: \(FileManager.default.currentDirectoryPath)")
			print("  2. File name is spelled correctly (case-sensitive)")
			print("  3. File is readable and contains text")
			print("\nTry: ls -la Journal_Total.txt")
			exit(1)
		}
		
		// For very large datasets, sample a subset
		if docs.count > maxSamples {
			print("\nDataset is large (\(docs.count) lines). Sampling \(maxSamples) lines for training...")
			print("(Use --samples <n> to change this, e.g., --samples 200000)")
			docs = Array(docs.shuffled().prefix(maxSamples))
		}
	}
	
	docs.shuffle()
	print("Loaded \(docs.count) documents")
	
	// Build vocabulary
	let allChars = Set(docs.joined())
	let uchars = allChars.sorted()
	let BOS = uchars.count
	let vocabSize = uchars.count + 1
	
	print("Vocabulary size: \(vocabSize)")
	print("Characters: \(String(uchars))")
	
	// Initialize parameters
	stateDict["wte"] = matrix(nout: vocabSize, nin: nEmbd)
	stateDict["wpe"] = matrix(nout: blockSize, nin: nEmbd)
	stateDict["lm_head"] = matrix(nout: vocabSize, nin: nEmbd)
	
	// 1) Multi-head attention block
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
	
	// Adam optimizer buffers
	var m = [Double](repeating: 0.0, count: params.count)
	var v = [Double](repeating: 0.0, count: params.count)
	let learningRate = 0.01
	let beta1 = 0.85
	let beta2 = 0.99
	let epsAdam = 1e-8
	
	// Training loop
	print("Training for \(numSteps) steps...")
	let startTime = Date()
	
	for step in 0..<numSteps {
		let doc = docs[step % docs.count]
		var tokens = [BOS]
		tokens.append(contentsOf: doc.map { ch in uchars.firstIndex(of: ch)! })
		tokens.append(BOS)
		
		let n = min(blockSize, tokens.count - 1)
		
		var keys = [[[Value]]](repeating: [], count: nLayer)
		var values = [[[Value]]](repeating: [], count: nLayer)
		var losses: [Value] = []
		
		for posId in 0..<n {
			let tokenId = tokens[posId]
			let targetId = tokens[posId + 1]
			let logits = gpt(tokenId: tokenId, posId: posId, keys: &keys, values: &values,
						   nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
			let probs = softmax(logits)
			let lossT = -probs[targetId].log()
			losses.append(lossT)
		}
		
		let loss = losses.reduce(Value(0), +) / Double(n)
		
		loss.backward()
		
		// Adam update
		let lrT = learningRate * (1.0 - Double(step) / Double(numSteps))
		for (idx, p) in params.enumerated() {
			m[idx] = beta1 * m[idx] + (1 - beta1) * p.grad
			v[idx] = beta2 * v[idx] + (1 - beta2) * p.grad * p.grad
			let mHat = m[idx] / (1 - pow(beta1, Double(step + 1)))
			let vHat = v[idx] / (1 - pow(beta2, Double(step + 1)))
			p.data -= lrT * mHat / (sqrt(vHat) + epsAdam)
			p.grad = 0.0
		}
		
		if (step + 1) % 100 == 0 || step == 0 {
			let now = Date()
			let elapsed = now.timeIntervalSince(startTime)
			let stepsPerSec = Double(step + 1) / elapsed
			let remaining = Double(numSteps - step - 1) / stepsPerSec
			let hours = Int(remaining) / 3600
			let mins = (Int(remaining) % 3600) / 60
			let eta = hours > 0 ? "\(hours)h\(mins)m" : "\(mins)m"
			
			print(String(format: "step %4d / %4d | loss %.4f | %.1f steps/s | ETA: %@", 
						step + 1, numSteps, loss.data, stepsPerSec, eta))
		}
	}
	
	// Save model
	print("\nSaving model to \(modelPath)...")
	var modelData = "\(vocabSize)\n"
	modelData += "\(String(uchars))\n"
	for (key, mat) in stateDict.sorted(by: { $0.key < $1.key }) {
		modelData += "\(key)\n"
		for row in mat {
			modelData += row.map { String($0.data) }.joined(separator: ",") + "\n"
		}
	}
	try? modelData.write(toFile: modelPath, atomically: true, encoding: .utf8)
	print("Model saved!")
	
} else if mode == "generate" {
	// Load model
	print("Loading model from \(modelPath)...")
	
	guard let modelContent = try? String(contentsOfFile: modelPath, encoding: .utf8) else {
		print("Error: Could not load model file. Train a model first!")
		exit(1)
	}
	
	let lines = modelContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
	let vocabSize = Int(lines[0])!
	let uchars = Array(lines[1])
	let BOS = uchars.count
	
	print("Vocabulary size: \(vocabSize)")
	
	// Determine matrix dimensions from model architecture
	let matrixSizes: [String: (Int, Int)] = [
		"wte": (vocabSize, nEmbd),
		"wpe": (blockSize, nEmbd),
		"lm_head": (vocabSize, nEmbd),
		"layer0.attn_wq": (nEmbd, nEmbd),
		"layer0.attn_wk": (nEmbd, nEmbd),
		"layer0.attn_wv": (nEmbd, nEmbd),
		"layer0.attn_wo": (nEmbd, nEmbd),
		"layer0.mlp_fc1": (4 * nEmbd, nEmbd),
		"layer0.mlp_fc2": (nEmbd, 4 * nEmbd),
		"layer1.attn_wq": (nEmbd, nEmbd),
		"layer1.attn_wk": (nEmbd, nEmbd),
		"layer1.attn_wv": (nEmbd, nEmbd),
		"layer1.attn_wo": (nEmbd, nEmbd),
		"layer1.mlp_fc1": (4 * nEmbd, nEmbd),
		"layer1.mlp_fc2": (nEmbd, 4 * nEmbd),
		"layer2.attn_wq": (nEmbd, nEmbd),
		"layer2.attn_wk": (nEmbd, nEmbd),
		"layer2.attn_wv": (nEmbd, nEmbd),
		"layer2.attn_wo": (nEmbd, nEmbd),
		"layer2.mlp_fc1": (4 * nEmbd, nEmbd),
		"layer2.mlp_fc2": (nEmbd, 4 * nEmbd),
		"layer3.attn_wq": (nEmbd, nEmbd),
		"layer3.attn_wk": (nEmbd, nEmbd),
		"layer3.attn_wv": (nEmbd, nEmbd),
		"layer3.attn_wo": (nEmbd, nEmbd),
		"layer3.mlp_fc1": (4 * nEmbd, nEmbd),
		"layer3.mlp_fc2": (nEmbd, 4 * nEmbd)
	]
	
	var lineIdx = 2
	while lineIdx < lines.count {
		let key = lines[lineIdx]
		lineIdx += 1
		
		guard let (nrows, _) = matrixSizes[key] else {
			print("Warning: Unknown key \(key)")
			continue
		}
		
		var mat: [[Value]] = []
		for _ in 0..<nrows {
			if lineIdx >= lines.count {
				break
			}
			let row = lines[lineIdx].split(separator: ",").map { Value(Double($0)!) }
			mat.append(row)
			lineIdx += 1
		}
		
		if mat.count == nrows {
			stateDict[key] = mat
		} else {
			print("Warning: Matrix \(key) has wrong size: got \(mat.count) rows, expected \(nrows)")
		}
	}
	
	print("Model loaded!\n")
	
	// Helper function to generate text from a prompt
	func generate(from promptText: String, numSamples: Int = 1) {
		for sampleIdx in 0..<numSamples {
			var keys = [[[Value]]](repeating: [], count: nLayer)
			var values = [[[Value]]](repeating: [], count: nLayer)
			
			var generated: [Character] = []
			var position = 0
			
			// First, process the prompt characters
			for ch in promptText {
				if let tokenId = uchars.firstIndex(of: ch) {
					if position < blockSize {
						_ = gpt(tokenId: tokenId, posId: position, keys: &keys, values: &values,
							   nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
						position += 1
					}
				}
			}
			
			// Now generate continuation
			var tokenId = BOS
			for posId in position..<blockSize {
				let logits = gpt(tokenId: tokenId, posId: posId, keys: &keys, values: &values,
							   nEmbd: nEmbd, nHead: nHead, nLayer: nLayer)
				let probs = softmax(logits.map { $0 / temperature })
				let weights = probs.map { $0.data }
				
				// Weighted random sampling
				let totalWeight = weights.reduce(0, +)
				var rand = Double.random(in: 0..<totalWeight)
				tokenId = 0
				for (idx, weight) in weights.enumerated() {
					rand -= weight
					if rand <= 0 {
						tokenId = idx
						break
					}
				}
				
				if tokenId == BOS {
					break
				}
				generated.append(uchars[tokenId])
			}
			
			if numSamples > 1 {
				print(String(format: "  %d: %@%@", sampleIdx + 1, promptText, String(generated)))
			} else {
				print("\(promptText)\(String(generated))")
			}
		}
	}
	
	// Interactive mode
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
		print("> ", terminator: "")
		fflush(stdout)
		
		guard let input = readLine() else {
			break
		}
		
		let trimmed = input.trimmingCharacters(in: .whitespaces)
		
		if trimmed.isEmpty {
			continue
		}
		
		// Check for commands
		if trimmed.lowercased() == "exit" || trimmed.lowercased() == "quit" {
			print("\nGoodbye! 👋")
			break
		}
		
		if trimmed.lowercased().hasPrefix("multi ") {
			if let n = Int(trimmed.dropFirst(6).trimmingCharacters(in: .whitespaces)) {
				numSamples = max(1, min(n, 20))
				print("✓ Will generate \(numSamples) variations for next prompt\n")
			} else {
				print("Usage: multi <number>\n")
			}
			continue
		}
		
		if trimmed.lowercased().hasPrefix("temp ") {
			if let t = Double(trimmed.dropFirst(5).trimmingCharacters(in: .whitespaces)) {
				temperature = max(0.1, min(t, 2.0))
				print("✓ Temperature set to \(temperature)\n")
			} else {
				print("Usage: temp <number> (e.g., temp 0.7)\n")
			}
			continue
		}
		
		// Generate from prompt
		generate(from: trimmed, numSamples: numSamples)
		print("")
		
		// Reset to single sample after multi command
		if numSamples > 1 {
			numSamples = 1
		}
	}
}

print("\nUsage:")
print("  Training:   ./microgpt --train file1.txt [--steps 1000] [--samples 100000] [--model model.txt]")
print("  Generation: ./microgpt --generate [--prompt 'text'] [--temp 0.5] [--model model.txt]")
print("              (Generation mode is now interactive by default)")
print("\nOptions:")
print("  --steps N     Number of training steps (default: 1000)")
print("  --samples N   Max lines to sample from dataset (default: 100000)")
print("  --temp N      Temperature 0.1-1.0 for generation (default: 0.5)")
print("  --model path  Path to save/load model (default: model.txt)")
