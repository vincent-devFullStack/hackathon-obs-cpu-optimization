import com.sun.management.OperatingSystemMXBean;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.lang.management.ThreadMXBean;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class CosineBenchmark {
    private static final class Dataset {
        final int n;
        final int m;
        final int d;
        final Path ePath;
        final Path aPath;
        final String datasetSha256;
        final double[] e;
        final double[] a;

        Dataset(int n, int m, int d, Path ePath, Path aPath, String datasetSha256, double[] e, double[] a) {
            this.n = n;
            this.m = m;
            this.d = d;
            this.ePath = ePath;
            this.aPath = aPath;
            this.datasetSha256 = datasetSha256;
            this.e = e;
            this.a = a;
        }
    }

    private static final class ProcSnapshot {
        final long threads;
        final long vmHwmKb;
        final long ctxVoluntary;
        final long ctxInvoluntary;
        final long minorFaults;
        final long majorFaults;

        ProcSnapshot(long threads, long vmHwmKb, long ctxVoluntary, long ctxInvoluntary, long minorFaults, long majorFaults) {
            this.threads = threads;
            this.vmHwmKb = vmHwmKb;
            this.ctxVoluntary = ctxVoluntary;
            this.ctxInvoluntary = ctxInvoluntary;
            this.minorFaults = minorFaults;
            this.majorFaults = majorFaults;
        }
    }

    private static String extractString(String json, String key) {
        Pattern p = Pattern.compile("\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
        Matcher m = p.matcher(json);
        if (!m.find()) {
            throw new IllegalArgumentException("missing key: " + key);
        }
        return m.group(1);
    }

    private static int extractInt(String json, String key) {
        Pattern p = Pattern.compile("\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*([0-9]+)");
        Matcher m = p.matcher(json);
        if (!m.find()) {
            throw new IllegalArgumentException("missing key: " + key);
        }
        return Integer.parseInt(m.group(1));
    }

    private static double[] readF64File(Path path, int expectedValues) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        int expectedBytes = expectedValues * 8;
        if (bytes.length != expectedBytes) {
            throw new IllegalArgumentException(
                    "bad size for " + path + ": got " + bytes.length + " expected " + expectedBytes
            );
        }
        double[] out = new double[expectedValues];
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < expectedValues; i++) {
            out[i] = bb.getDouble();
        }
        return out;
    }

    private static String sha256Concat(Path a, Path b) throws Exception {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        md.update(Files.readAllBytes(a));
        md.update(Files.readAllBytes(b));
        byte[] digest = md.digest();
        StringBuilder sb = new StringBuilder();
        for (byte bt : digest) {
            sb.append(String.format(Locale.US, "%02x", bt));
        }
        return sb.toString();
    }

    private static Dataset loadDataset(Path metadataPath) throws Exception {
        String json = Files.readString(metadataPath);

        String format = extractString(json, "format");
        String dtype = extractString(json, "dtype");
        if (!"cosine-benchmark-v1".equals(format)) {
            throw new IllegalArgumentException("unsupported format: " + format);
        }
        if (!"float64-le".equals(dtype)) {
            throw new IllegalArgumentException("unsupported dtype: " + dtype);
        }

        int n = extractInt(json, "N");
        int m = extractInt(json, "M");
        int d = extractInt(json, "D");
        String eFile = extractString(json, "E_file");
        String aFile = extractString(json, "A_file");

        Path base = metadataPath.getParent();
        if (base == null) {
            base = Paths.get(".");
        }

        Path ePath = Paths.get(eFile);
        if (!ePath.isAbsolute()) {
            ePath = base.resolve(ePath);
        }

        Path aPath = Paths.get(aFile);
        if (!aPath.isAbsolute()) {
            aPath = base.resolve(aPath);
        }

        String datasetSha = sha256Concat(ePath, aPath);
        double[] e = readF64File(ePath, n * d);
        double[] a = readF64File(aPath, m * d);

        return new Dataset(n, m, d, ePath, aPath, datasetSha, e, a);
    }

    private static double cosineAllPairsChecksum(Dataset ds) {
        double[] axisNorms = new double[ds.m];
        for (int j = 0; j < ds.m; j++) {
            int aBase = j * ds.d;
            double s = 0.0;
            for (int k = 0; k < ds.d; k++) {
                double x = ds.a[aBase + k];
                s += x * x;
            }
            axisNorms[j] = Math.sqrt(s);
        }

        double checksum = 0.0;
        for (int i = 0; i < ds.n; i++) {
            int eBase = i * ds.d;
            double embNormSq = 0.0;
            for (int k = 0; k < ds.d; k++) {
                double x = ds.e[eBase + k];
                embNormSq += x * x;
            }
            double embNorm = Math.sqrt(embNormSq);
            if (embNorm == 0.0) {
                continue;
            }

            for (int j = 0; j < ds.m; j++) {
                int aBase = j * ds.d;
                double dot = 0.0;
                for (int k = 0; k < ds.d; k++) {
                    dot += ds.e[eBase + k] * ds.a[aBase + k];
                }
                double denom = embNorm * axisNorms[j];
                if (denom != 0.0) {
                    checksum += dot / denom;
                }
            }
        }

        return checksum;
    }

    private static long cpuTimeNs() {
        OperatingSystemMXBean osMx = ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);
        if (osMx != null) {
            long v = osMx.getProcessCpuTime();
            if (v >= 0L) {
                return v;
            }
        }

        ThreadMXBean threadMx = ManagementFactory.getThreadMXBean();
        if (threadMx.isCurrentThreadCpuTimeSupported()) {
            return threadMx.getCurrentThreadCpuTime();
        }

        return -1L;
    }

    private static ProcSnapshot readProcSnapshot() {
        long threads = -1L;
        long vmHwmKb = -1L;
        long ctxVol = -1L;
        long ctxInvol = -1L;
        long minflt = -1L;
        long majflt = -1L;

        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("Threads:")) {
                    threads = Long.parseLong(line.substring(line.indexOf(':') + 1).trim());
                } else if (line.startsWith("VmHWM:")) {
                    String[] parts = line.substring(line.indexOf(':') + 1).trim().split("\\s+");
                    vmHwmKb = Long.parseLong(parts[0]);
                } else if (line.startsWith("voluntary_ctxt_switches:")) {
                    ctxVol = Long.parseLong(line.substring(line.indexOf(':') + 1).trim());
                } else if (line.startsWith("nonvoluntary_ctxt_switches:")) {
                    ctxInvol = Long.parseLong(line.substring(line.indexOf(':') + 1).trim());
                }
            }
        } catch (Exception ignored) {
            // best effort
        }

        try {
            String stat = Files.readString(Paths.get("/proc/self/stat")).trim();
            int rparen = stat.lastIndexOf(')');
            if (rparen > 0 && rparen + 2 < stat.length()) {
                String tail = stat.substring(rparen + 2);
                String[] f = tail.split("\\s+");
                if (f.length > 9) {
                    minflt = Long.parseLong(f[7]);
                    majflt = Long.parseLong(f[9]);
                }
            }
        } catch (Exception ignored) {
            // best effort
        }

        return new ProcSnapshot(threads, vmHwmKb, ctxVol, ctxInvol, minflt, majflt);
    }

    private static String jsonEscape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String jsonArrayOfStrings(java.util.List<String> items) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("\"").append(jsonEscape(items.get(i))).append("\"");
        }
        sb.append("]");
        return sb.toString();
    }

    private static int parseIntArg(String[] args, int i, String name) {
        if (i + 1 >= args.length) {
            throw new IllegalArgumentException("missing value for " + name);
        }
        return Integer.parseInt(args[i + 1]);
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);

        String metadata = "data/metadata.json";
        String expectedDatasetSha = "";
        int warmup = 5;
        int runs = 30;
        int repeat = 50;
        boolean selfCheck = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--metadata":
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("missing value for --metadata");
                    }
                    metadata = args[++i];
                    break;
                case "--warmup":
                    warmup = parseIntArg(args, i, "--warmup");
                    i++;
                    break;
                case "--runs":
                    runs = parseIntArg(args, i, "--runs");
                    i++;
                    break;
                case "--repeat":
                    repeat = parseIntArg(args, i, "--repeat");
                    i++;
                    break;
                case "--expected-dataset-sha":
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("missing value for --expected-dataset-sha");
                    }
                    expectedDatasetSha = args[++i];
                    break;
                case "--self-check":
                    selfCheck = true;
                    break;
                default:
                    throw new IllegalArgumentException(
                            "Usage: java CosineBenchmark [--metadata PATH] [--warmup N] [--runs N] [--repeat N] [--expected-dataset-sha SHA] [--self-check]"
                    );
            }
        }

        if (warmup < 0 || runs <= 0 || repeat <= 0) {
            throw new IllegalArgumentException("warmup must be >= 0, runs and repeat must be > 0");
        }

        Dataset ds = loadDataset(Paths.get(metadata));
        if (!expectedDatasetSha.isEmpty() && !ds.datasetSha256.equals(expectedDatasetSha)) {
            throw new IllegalArgumentException(
                    "dataset sha mismatch: expected " + expectedDatasetSha + " got " + ds.datasetSha256
            );
        }

        RuntimeMXBean runtimeMx = ManagementFactory.getRuntimeMXBean();
        java.util.List<String> jvmArgs = runtimeMx.getInputArguments();
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        System.err.println("[JAVA] jvm_input_args=" + jvmArgs);
        System.err.println("[JAVA] available_processors=" + availableProcessors);

        if (selfCheck) {
            double checksum = cosineAllPairsChecksum(ds);
            System.out.println(String.format(
                    Locale.US,
                    "{\"type\":\"self_check\",\"impl\":\"java-naive\",\"ok\":true,\"N\":%d,\"M\":%d,\"D\":%d,\"dataset_sha256\":\"%s\",\"checksum\":%.17g}",
                    ds.n,
                    ds.m,
                    ds.d,
                    ds.datasetSha256,
                    checksum
            ));
            return;
        }

        String runtime = String.format(
                Locale.US,
                "{\"java_version\":\"%s\",\"java_vm\":\"%s\",\"warmup_executed\":%d,\"available_processors\":%d,\"jvm_input_args\":%s}",
                jsonEscape(System.getProperty("java.version", "unknown")),
                jsonEscape(System.getProperty("java.vm.name", "unknown")),
                warmup,
                availableProcessors,
                jsonArrayOfStrings(jvmArgs)
        );

        System.out.println(String.format(
                Locale.US,
                "{\"type\":\"meta\",\"impl\":\"java-naive\",\"N\":%d,\"M\":%d,\"D\":%d,\"repeat\":%d,\"warmup\":%d,\"runs\":%d,\"dataset_sha256\":\"%s\",\"build_flags\":\"\",\"runtime\":%s}",
                ds.n,
                ds.m,
                ds.d,
                repeat,
                warmup,
                runs,
                ds.datasetSha256,
                runtime
        ));

        for (int i = 0; i < warmup; i++) {
            cosineAllPairsChecksum(ds);
        }

        for (int runId = 0; runId < runs; runId++) {
            ProcSnapshot before = readProcSnapshot();
            long t0 = System.nanoTime();
            long c0 = cpuTimeNs();

            double checksumAcc = 0.0;
            for (int r = 0; r < repeat; r++) {
                checksumAcc += cosineAllPairsChecksum(ds);
            }

            long t1 = System.nanoTime();
            long c1 = cpuTimeNs();
            ProcSnapshot after = readProcSnapshot();

            long wallNs = (t1 - t0) / repeat;
            long cpuNs = (c0 >= 0 && c1 >= 0) ? ((c1 - c0) / repeat) : -1L;

            long ctxVol = (before.ctxVoluntary >= 0 && after.ctxVoluntary >= 0)
                    ? (after.ctxVoluntary - before.ctxVoluntary)
                    : -1L;
            long ctxInvol = (before.ctxInvoluntary >= 0 && after.ctxInvoluntary >= 0)
                    ? (after.ctxInvoluntary - before.ctxInvoluntary)
                    : -1L;
            long minflt = (before.minorFaults >= 0 && after.minorFaults >= 0)
                    ? (after.minorFaults - before.minorFaults)
                    : -1L;
            long majflt = (before.majorFaults >= 0 && after.majorFaults >= 0)
                    ? (after.majorFaults - before.majorFaults)
                    : -1L;
            long maxThreads = Math.max(before.threads, after.threads);

            String line = String.format(
                    Locale.US,
                    "{\"type\":\"run\",\"impl\":\"java-naive\",\"run_id\":%d,\"wall_ns\":%d,\"cpu_ns\":%d,\"checksum\":%.17g,\"max_rss_kb\":%d,\"ctx_voluntary\":%d,\"ctx_involuntary\":%d,\"minor_faults\":%d,\"major_faults\":%d,\"max_threads\":%d}",
                    runId,
                    wallNs,
                    cpuNs,
                    checksumAcc,
                    after.vmHwmKb,
                    ctxVol,
                    ctxInvol,
                    minflt,
                    majflt,
                    maxThreads
            );
            System.out.println(line);
        }
    }
}
