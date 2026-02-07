package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type Metadata struct {
	Format string `json:"format"`
	DType  string `json:"dtype"`
	N      int    `json:"N"`
	M      int    `json:"M"`
	D      int    `json:"D"`
	EFile  string `json:"E_file"`
	AFile  string `json:"A_file"`
}

type Dataset struct {
	N             int
	M             int
	D             int
	E             []float64
	A             []float64
	DatasetSHA256 string
}

type ProcSnapshot struct {
	Threads        int
	MaxRssKB       int64
	CtxVoluntary   int64
	CtxInvoluntary int64
	MinorFaults    int64
	MajorFaults    int64
}

func readMetadata(path string) (*Metadata, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Metadata
	if err := json.Unmarshal(buf, &m); err != nil {
		return nil, err
	}
	if m.Format != "cosine-benchmark-v1" {
		return nil, fmt.Errorf("unsupported format: %s", m.Format)
	}
	if m.DType != "float64-le" {
		return nil, fmt.Errorf("unsupported dtype: %s", m.DType)
	}
	if m.N <= 0 || m.M <= 0 || m.D <= 0 {
		return nil, errors.New("invalid N/M/D")
	}
	if m.EFile == "" || m.AFile == "" {
		return nil, errors.New("missing E_file/A_file")
	}
	return &m, nil
}

func readF64File(path string, expected int) ([]float64, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(buf) != expected*8 {
		return nil, fmt.Errorf("bad size for %s: got %d expected %d", path, len(buf), expected*8)
	}
	out := make([]float64, expected)
	for i := 0; i < expected; i++ {
		off := i * 8
		bits := binary.LittleEndian.Uint64(buf[off : off+8])
		out[i] = math.Float64frombits(bits)
	}
	return out, nil
}

func sha256Concat(paths ...string) (string, error) {
	h := sha256.New()
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return "", err
		}
		_, err = io.Copy(h, f)
		_ = f.Close()
		if err != nil {
			return "", err
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func loadDataset(metadataPath string) (*Dataset, error) {
	meta, err := readMetadata(metadataPath)
	if err != nil {
		return nil, err
	}
	base := filepath.Dir(metadataPath)
	ePath := meta.EFile
	if !filepath.IsAbs(ePath) {
		ePath = filepath.Join(base, ePath)
	}
	aPath := meta.AFile
	if !filepath.IsAbs(aPath) {
		aPath = filepath.Join(base, aPath)
	}

	datasetSHA, err := sha256Concat(ePath, aPath)
	if err != nil {
		return nil, err
	}

	e, err := readF64File(ePath, meta.N*meta.D)
	if err != nil {
		return nil, err
	}
	a, err := readF64File(aPath, meta.M*meta.D)
	if err != nil {
		return nil, err
	}

	return &Dataset{N: meta.N, M: meta.M, D: meta.D, E: e, A: a, DatasetSHA256: datasetSHA}, nil
}

func normalizeAxes(ds *Dataset) []float64 {
	normed := make([]float64, ds.M*ds.D)
	for j := 0; j < ds.M; j++ {
		aBase := j * ds.D
		row := ds.A[aBase : aBase+ds.D]
		normSq := 0.0
		for _, x := range row {
			normSq += x * x
		}
		norm := math.Sqrt(normSq)
		inv := 0.0
		if norm != 0.0 {
			inv = 1.0 / norm
		}
		for k := 0; k < ds.D; k++ {
			normed[aBase+k] = row[k] * inv
		}
	}
	return normed
}

func cosineAllPairsChecksumOpt(ds *Dataset, aNormed []float64) float64 {
	checksum := 0.0
	for i := 0; i < ds.N; i++ {
		eBase := i * ds.D
		eRow := ds.E[eBase : eBase+ds.D]

		embNormSq := 0.0
		for _, x := range eRow {
			embNormSq += x * x
		}
		embNorm := math.Sqrt(embNormSq)
		if embNorm == 0.0 {
			continue
		}
		invEmbNorm := 1.0 / embNorm

		for j := 0; j < ds.M; j++ {
			aBase := j * ds.D
			aRow := aNormed[aBase : aBase+ds.D]
			dot := 0.0
			for k := 0; k < ds.D; k++ {
				dot += eRow[k] * aRow[k]
			}
			checksum += dot * invEmbNorm
		}
	}
	return checksum
}

func cpuTimeNs() int64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return -1
	}
	userNs := ru.Utime.Sec*1_000_000_000 + ru.Utime.Usec*1_000
	sysNs := ru.Stime.Sec*1_000_000_000 + ru.Stime.Usec*1_000
	return userNs + sysNs
}

func procThreads() int {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return -1
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "Threads:") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				v, err := strconv.Atoi(parts[1])
				if err == nil {
					return v
				}
			}
		}
	}
	return -1
}

func procSnapshot() ProcSnapshot {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return ProcSnapshot{Threads: procThreads(), MaxRssKB: -1, CtxVoluntary: -1, CtxInvoluntary: -1, MinorFaults: -1, MajorFaults: -1}
	}
	return ProcSnapshot{
		Threads:        procThreads(),
		MaxRssKB:       ru.Maxrss,
		CtxVoluntary:   ru.Nvcsw,
		CtxInvoluntary: ru.Nivcsw,
		MinorFaults:    ru.Minflt,
		MajorFaults:    ru.Majflt,
	}
}

func delta(after int64, before int64) int64 {
	if after < 0 || before < 0 {
		return -1
	}
	return after - before
}

func main() {
	metadata := flag.String("metadata", "data/metadata.json", "Path to metadata JSON")
	warmup := flag.Int("warmup", 5, "Warmup iterations")
	runs := flag.Int("runs", 30, "Benchmark runs")
	repeat := flag.Int("repeat", 50, "Repeat per run")
	expectedDatasetSHA := flag.String("expected-dataset-sha", "", "Expected dataset checksum")
	selfCheck := flag.Bool("self-check", false, "Run only one checksum and exit")
	flag.Parse()

	if *warmup < 0 || *runs <= 0 || *repeat <= 0 {
		fmt.Fprintln(os.Stderr, "warmup must be >= 0, runs and repeat must be > 0")
		os.Exit(1)
	}

	ds, err := loadDataset(*metadata)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load dataset: %v\n", err)
		os.Exit(1)
	}

	if *expectedDatasetSHA != "" && ds.DatasetSHA256 != *expectedDatasetSHA {
		fmt.Fprintf(os.Stderr, "dataset sha mismatch: expected %s got %s\n", *expectedDatasetSHA, ds.DatasetSHA256)
		os.Exit(1)
	}

	aNormed := normalizeAxes(ds)

	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)

	if *selfCheck {
		checksum := cosineAllPairsChecksumOpt(ds, aNormed)
		rec := map[string]interface{}{
			"type":           "self_check",
			"impl":           "go-opt",
			"ok":             true,
			"N":              ds.N,
			"M":              ds.M,
			"D":              ds.D,
			"dataset_sha256": ds.DatasetSHA256,
			"checksum":       checksum,
			"runtime": map[string]interface{}{
				"gomaxprocs":     runtime.GOMAXPROCS(0),
				"gomaxprocs_env": os.Getenv("GOMAXPROCS"),
				"num_cpu":        runtime.NumCPU(),
				"go_version":     runtime.Version(),
			},
		}
		_ = enc.Encode(rec)
		return
	}

	meta := map[string]interface{}{
		"type":           "meta",
		"impl":           "go-opt",
		"N":              ds.N,
		"M":              ds.M,
		"D":              ds.D,
		"repeat":         *repeat,
		"warmup":         *warmup,
		"runs":           *runs,
		"dataset_sha256": ds.DatasetSHA256,
		"build_flags":    "",
		"runtime": map[string]interface{}{
			"go_version":      runtime.Version(),
			"goos":            runtime.GOOS,
			"goarch":          runtime.GOARCH,
			"gomaxprocs":      runtime.GOMAXPROCS(0),
			"gomaxprocs_env":  os.Getenv("GOMAXPROCS"),
			"num_cpu":         runtime.NumCPU(),
			"warmup_executed": *warmup,
		},
	}
	_ = enc.Encode(meta)

	for i := 0; i < *warmup; i++ {
		_ = cosineAllPairsChecksumOpt(ds, aNormed)
	}

	for runID := 0; runID < *runs; runID++ {
		before := procSnapshot()
		t0 := time.Now()
		c0 := cpuTimeNs()

		checksumAcc := 0.0
		for r := 0; r < *repeat; r++ {
			checksumAcc += cosineAllPairsChecksumOpt(ds, aNormed)
		}

		elapsed := time.Since(t0).Nanoseconds()
		c1 := cpuTimeNs()
		after := procSnapshot()

		wallNs := elapsed / int64(*repeat)
		cpuNs := int64(-1)
		if c0 >= 0 && c1 >= 0 {
			cpuNs = (c1 - c0) / int64(*repeat)
		}

		maxThreads := before.Threads
		if after.Threads > maxThreads {
			maxThreads = after.Threads
		}

		rec := map[string]interface{}{
			"type":            "run",
			"impl":            "go-opt",
			"run_id":          runID,
			"wall_ns":         wallNs,
			"cpu_ns":          cpuNs,
			"checksum":        checksumAcc,
			"max_rss_kb":      after.MaxRssKB,
			"ctx_voluntary":   delta(after.CtxVoluntary, before.CtxVoluntary),
			"ctx_involuntary": delta(after.CtxInvoluntary, before.CtxInvoluntary),
			"minor_faults":    delta(after.MinorFaults, before.MinorFaults),
			"major_faults":    delta(after.MajorFaults, before.MajorFaults),
			"max_threads":     maxThreads,
		}
		_ = enc.Encode(rec)
	}
}
