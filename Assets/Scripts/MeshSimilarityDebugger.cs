using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using UnityEngine;
using System.Linq;


#if UNITY_EDITOR
using UnityEditor;
#endif

// Inspector'da salt okunur alan:
public class ReadOnlyAttribute : PropertyAttribute { }

// NN çizgilerini güvenle tutmak için:
public struct NNLine
{
    public Vector3 from;
    public Vector3 to;
    public NNLine(Vector3 f, Vector3 t) { from = f; to = t; }
}

public enum SimilarityMetric { ChamferMean, TrimmedHausdorff }

// --- Hole-aware için ek enum ---
enum BoundaryClass { Outer, Hole, Unknown }

// Kenar segmenti (delik/outer sınıfıyla)
struct Seg
{
    public Vector3 a, b;
    public BoundaryClass cls;
    public Seg(Vector3 a, Vector3 b, BoundaryClass c) { this.a = a; this.b = b; this.cls = c; }
}

[ExecuteAlways]
public class MeshSimilarityDebugger : MonoBehaviour
{
    [Header("Mode")]
    [Tooltip("true: her çocuğu ayrı değerlendir, skorları eşit ağırlıkla ortala. false: tüm noktaları tek bulutta hesapla.")]
    public bool perChildMode = false; // 3D global default

    // MeshSimilarityDebugger.cs
    [Header("Per-Child Options")]
    public bool childCenterAlignByBounds = true;  // child bazında pozisyonu yok say

    [Header("Robustness (Global veya Childwise Chamfer için)")]
    [Tooltip("En büyük yüzde kaçlık mesafeyi atacağız (0..0.5). Örn: 0.10 = %10")]
    [Range(0f, 0.5f)] public float trimTopPercent = 0.10f;

    [Tooltip("Mesafeleri şu oran * diag ile tavanla. Örn: 0.10 = diag’ın %10’u")]
    [Range(0f, 1f)] public float capAtDiagFrac = 0.10f;

    [Tooltip("Eşleşmiş sayılacak mesafe eşiği (oran*diag). Örn: 0.02 = %2")]
    [Range(0f, 0.2f)] public float matchDeltaFrac = 0.005f;

    public bool useTrimmed = false;
    public bool useCapped = false;

    [Header("Targets (Inspector’dan atayın)")]
    public GameObject objectA;
    public GameObject objectB;

    [Header("Sampling")]
    [Range(100, 50000)] public int samplesPerMesh = 12000;
    [Tooltip("true: yalnızca şekil (grubun kök local uzayı); false: world poz/rot/scale etkili")]
    public bool poseInvariant = true;
    [Tooltip("Deterministiklik için sabit tohum")]
    public int randomSeed = 12345;

    [Header("Feature-aware sampling")]
    [Tooltip("Kenar ve deliklerin etkisini artır (boundary örnekleme).")]
    public bool emphasizeEdges = true;
    [Range(0f, 1f)] public float edgePortion = 0.8f;

    [Header("Alignment helpers (global mod)")]
    public bool centerAlignByBounds = true; // dış kutu merkezlerini çakıştır
    public bool projectPlanar = false;      // düz paneller için Z'yi sıfırla

    [Header("Metric")]
    public SimilarityMetric metric = SimilarityMetric.TrimmedHausdorff;
    [Range(0.5f, 0.999f)] public float hausdorffPercentile = 0.995f;










    [Header("Hole-aware (2D)")]
    public bool holeAwareMatching = false; // 3D globalde kapalı
    [Range(0f, 4f)] public float holeEdgeWeight = 2f;
    [Range(1f, 5f)] public float holePenaltyMultiplier = 2f;
    public bool hausdorffBoundaryOnly = false;
    public bool forcePlanarFor2D = false;


    [Header("Hole W/H Similarity (2D)")]
    public bool useHoleWHPenalty = true;                  // W,H yakınlığı (0..1 ceza → 1-penalty = sim)
    [Range(0f, 5f)] public float holeWHPenaltyLambda = 1.2f;

    [Tooltip("p-norm üssü; 1=yumuşak, 3+=agresif")]
    [Range(1f, 4f)] public float holeWHExp = 2.4f;

    [Tooltip("Büyüklüğe göre sıralı deliklerin yüzde kaçını dikkate alalım")]
    [Range(0f, 1f)] public float holeWHTopFrac = 0.85f;

    [Tooltip("Büyük deliklere rank ağırlığı uygula")]
    public bool holeWHRankWeighted = true;

    [Header("Hole Count Penalty")]
    public bool useHoleCountOnly = true;                  // sadece sayıya dayalı ek ceza
    [Range(0f, 5f)] public float holeCountPenaltyLambda = 0.8f;














    [Header("Similarity (Chamfer-based)")]
    [Tooltip("0..1: Similarity = 1 - (Chamfer/Diagonal)")]
    [ReadOnly] public float geometrySimilarity01;
    [ReadOnly] public float chamferDistance;      // mutlak Chamfer veya yüzdelik NN
    [ReadOnly] public float bboxDiagonal;         // normalize için diag

    [Header("Debug & Control")]
    public bool autoComputeOnPlay = true;
    public KeyCode recomputeKey = KeyCode.R;

    [Header("Gizmos")]
    public bool drawGizmos = true;
    public bool drawWhenUnselected = false;
    [Range(0.001f, 0.1f)] public float pointSize = 0.01f;
    public bool drawNearestNeighborLines = false;
    [Range(10, 3000)] public int maxNNLinesPerSet = 200;

    [Header("Optional: Instantiate point markers (runtime)")]
    public bool spawnPointMarkers = false;
    public GameObject pointPrefab; // küçük bir sphere prefab (isteğe bağlı)
    public Transform markersRoot;

    // --- Rotasyon cezası/benzerliği (Per-Child içinde sade) ---
    public enum RotationReference { RootRelative, World, Local }
    [Header("Rotation Penalty (Per-Child)")]
    public bool useNormalPenalty = true;        // (şekil için) Normale dayalı ceza
    public float normalPenaltyLambda = 1.5f;    // Normal farkının etkisi

    public bool useRotationPenalty = true;      // Çocukların rotasyon benzerliği
    public RotationReference rotationReference = RotationReference.RootRelative;

    public enum RotationPenaltyMode { RotationOnly, Multiplicative, LinearBlend, MaxOfBoth }
    [Tooltip("RotationOnly: sadece açı benzerliğini kullan (sade). Diğerleri şekil ile birleştirir.")]
    public RotationPenaltyMode rotMode = RotationPenaltyMode.RotationOnly;

    [Range(0.5f, 6f)] public float rotEmphasis = 2.5f;
    [Range(1f, 6f)] public float rotGamma = 3f;
    [Range(5f, 180f)] public float rotCapDeg = 30f;
    [Range(0f, 1f)] public float rotWeight = 0.7f;

    // --- Position penalty (rotasyonla birlikte) ---
    public enum PositionReference { RootRelative, World, Local }
    public enum PoseCombine { Multiplicative, Min, LinearBlend }

    [Header("Position Penalty (Per-Child)")]
    public bool usePositionPenalty = true;
    public PositionReference positionReference = PositionReference.RootRelative;
    [Tooltip("Mesafe tavanı (m). Bu mesafe sonrası posSim hızla 0'a yaklaşır.")]
    [Range(0.001f, 10f)] public float posCap = 0.05f; // 5 cm
    [Range(1f, 6f)] public float posGamma = 3f;
    [Range(0.5f, 6f)] public float posEmphasis = 3f;
    [Tooltip("rotSim ile posSim nasıl birleşsin?")]
    public PoseCombine poseCombine = PoseCombine.Min;
    [Range(0f, 1f)] public float poseBlendWeight = 0.5f; // LinearBlend için

    public enum ChildAgg { Mean, GeometricMean, Min, Pct25 }
    [Header("Per-Child Aggregation")]
    public ChildAgg childAggregation = ChildAgg.Mean;

    // --- Global toplama ve agresiflik üssü ---
    public enum GlobalAgg { Mean, Max, GeometricMean }
    [Header("Global Aggregation")]
    public GlobalAgg globalAggregation = GlobalAgg.Max;

    [Header("Aggressiveness")]
    [Tooltip("Mesafeleri üssü alarak uzak farkları büyütür (1=normal, 2–3=agresif).")]
    [Range(1f, 4f)] public float distanceExponent = 2.5f;

    public enum DiagNorm { Union, Mean, Min }
    [Header("Normalization")]
    public DiagNorm diagonalNorm = DiagNorm.Min;

    public enum HausdorffAgg { MeanOfDirs, MaxOfDirs }
    [Header("Hausdorff Aggregation")]
    public HausdorffAgg hausdorffAggregation = HausdorffAgg.MaxOfDirs;

    [Header("Scale penalty")]
    public bool useScalePenalty = true;
    [Range(0f, 2f)] public float scalePenaltyLambda = 0.6f;
    public Vector3 axisPenaltyWeights = new Vector3(1f, 1.5f, 1f);

    [Header("Post-sharpen")]
    [Range(1f, 6f)] public float simSharpnessPow = 2.5f;

    string modeStr = "Global";
    string recallStr = "-";
    string f1Str = "-";

    [Header("Match Stats (Read-only)")]
    [ReadOnly] public float matchRecall;
    [ReadOnly] public float matchPrecision;
    [ReadOnly] public float matchF1;

    // Internal state (gizmos)
    private readonly List<Vector3> ptsA = new List<Vector3>();
    private readonly List<Vector3> ptsB = new List<Vector3>();
    private readonly List<NNLine> nnLinesA2B = new List<NNLine>();
    private readonly List<NNLine> nnLinesB2A = new List<NNLine>();

    // Hole-aware: her noktanın sınıf etiketi (Outer/Hole/Unknown)
    private readonly List<BoundaryClass> clsA = new List<BoundaryClass>();
    private readonly List<BoundaryClass> clsB = new List<BoundaryClass>();

    [ReadOnly] public float recallAB_last;
    [ReadOnly] public float recallBA_last;


    // --- lifecycle ---
    void Start()
    {
        if (Application.isPlaying && autoComputeOnPlay) ComputeAll();
    }
    void Update()
    {
        if (Input.GetKeyDown(recomputeKey)) ComputeAll();
    }

    // --- helpers ---
    Vector3 Centroid(List<Vector3> pts)
    {
        if (pts == null || pts.Count == 0) return Vector3.zero;
        Vector3 s = Vector3.zero; for (int i = 0; i < pts.Count; i++) s += pts[i];
        return s / pts.Count;
    }

    float RotationSimilarityFromTheta(float thetaDeg)
    {
        float t = Mathf.Clamp01(thetaDeg / Mathf.Max(1e-3f, rotCapDeg));
        float rotSim = 1f - Mathf.Pow(t, Mathf.Max(1f, rotGamma));
        rotSim = Mathf.Pow(Mathf.Clamp01(rotSim), Mathf.Max(0.5f, rotEmphasis));
        return Mathf.Clamp01(rotSim);
    }

    // --- Position helpers ---
    Vector3 GetRefPosition(Transform child, Transform root)
    {
        switch (positionReference)
        {
            case PositionReference.World: return child.position;
            case PositionReference.Local: return child.localPosition;
            case PositionReference.RootRelative:
            default: return root.InverseTransformPoint(child.position);
        }
    }

    float PositionSimilarityFromDelta(float delta)
    {
        float t = Mathf.Clamp01(delta / Mathf.Max(1e-6f, posCap));
        float s = 1f - Mathf.Pow(t, Mathf.Max(1f, posGamma));
        s = Mathf.Pow(Mathf.Clamp01(s), Mathf.Max(0.5f, posEmphasis));
        return Mathf.Clamp01(s);
    }

    void CenterAlignByBoundsPts(List<Vector3> A, List<Vector3> B)
    {
        if (A == null || B == null || A.Count == 0 || B.Count == 0) return;
        var bA = BoundsFromPoints(A);
        var bB = BoundsFromPoints(B);
        Vector3 ca = bA.center, cb = bB.center;
        for (int i = 0; i < A.Count; i++) A[i] -= ca;
        for (int i = 0; i < B.Count; i++) B[i] -= cb;
    }

    void ProjectToBestPlane(List<Vector3> a, List<Vector3> b)
    {
        int n = (a?.Count ?? 0) + (b?.Count ?? 0);
        if (n < 3) return;

        var all = new List<Vector3>(n);
        all.AddRange(a); all.AddRange(b);
        var c = Centroid(all);

        double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
        for (int i = 0; i < all.Count; i++)
        {
            var p = all[i] - c;
            xx += p.x * p.x; xy += p.x * p.y; xz += p.x * p.z;
            yy += p.y * p.y; yz += p.y * p.z; zz += p.z * p.z;
        }

        Vector3 ex = new Vector3(1f, 0f, (float)((xx > 1e-9) ? xz / xx : 0)); ex.Normalize();
        Vector3 ey = new Vector3(0f, 1f, (float)((yy > 1e-9) ? yz / yy : 0));
        ey = (ey - Vector3.Dot(ey, ex) * ex).normalized;
        if (ey.sqrMagnitude < 1e-6f) ey = Vector3.Cross(ex, Vector3.up).normalized;

        Vector3 ez = Vector3.Cross(ex, ey).normalized;
        Vector3 u = ex;
        Vector3 v = Vector3.Cross(ez, u).normalized;

        for (int i = 0; i < a.Count; i++)
        {
            var p = a[i] - c;
            a[i] = new Vector3(Vector3.Dot(p, u), Vector3.Dot(p, v), 0f);
        }
        for (int i = 0; i < b.Count; i++)
        {
            var p = b[i] - c;
            b[i] = new Vector3(Vector3.Dot(p, u), Vector3.Dot(p, v), 0f);
        }
    }

    // --- Kenar toplama: boundary edge'leri çıkar (outer + hole)
    static void CollectBoundarySegments(Mesh m, Transform tr, Transform root, bool poseInv, List<Seg> outSegs)
    {
        var tris = m.triangles; var verts = m.vertices;
        var edgeCount = new Dictionary<(int, int), int>();

        for (int i = 0; i < tris.Length; i += 3)
        {
            int i0 = tris[i], i1 = tris[i + 1], i2 = tris[i + 2];
            void add(int a, int b)
            {
                var k = a < b ? (a, b) : (b, a);
                edgeCount.TryGetValue(k, out int c); edgeCount[k] = c + 1;
            }
            add(i0, i1); add(i1, i2); add(i2, i0);
        }

        outSegs.Clear();
        foreach (var kv in edgeCount)
        {
            if (kv.Value == 1)
            {
                var (i0, i1) = kv.Key;
                var p0 = poseInv ? root.InverseTransformPoint(tr.TransformPoint(verts[i0])) : tr.TransformPoint(verts[i0]);
                var p1 = poseInv ? root.InverseTransformPoint(tr.TransformPoint(verts[i1])) : tr.TransformPoint(verts[i1]);
                outSegs.Add(new Seg(p0, p1, BoundaryClass.Unknown));
            }
        }
    }

    static void SampleSegments(List<Seg> segs, int count, List<Vector3> outPts)
    {
        if (count <= 0 || segs.Count == 0) return;
        var lens = new List<float>(segs.Count); float total = 0f;
        for (int i = 0; i < segs.Count; i++) { float L = (segs[i].a - segs[i].b).magnitude; lens.Add(L); total += L; }
        for (int k = 0; k < count; k++)
        {
            float r = UnityEngine.Random.value * total; int idx = 0; float acc = 0;
            for (; idx < segs.Count; idx++) { acc += lens[idx]; if (r <= acc) break; }
            idx = Mathf.Clamp(idx, 0, segs.Count - 1);
            float t = UnityEngine.Random.value;
            outPts.Add(Vector3.Lerp(segs[idx].a, segs[idx].b, t));
        }
    }

    // --- Döngü kurma ve sınıflandırma (outer vs hole) ---
    static List<List<Vector3>> BuildLoopsFromSegments(List<Seg> segs, float snapEps = 1e-4f)
    {
        var loops = new List<List<Vector3>>();
        var map = new Dictionary<Vector3, List<int>>(new Vec3KeyEq(snapEps));
        for (int i = 0; i < segs.Count; i++)
        {
            Add(map, segs[i].a, i);
            Add(map, segs[i].b, i);
        }
        var used = new bool[segs.Count];

        for (int i = 0; i < segs.Count; i++)
        {
            if (used[i]) continue;
            var loop = new List<Vector3>();
            int cur = i; Vector3 start = segs[cur].a; Vector3 target = segs[cur].b;
            used[cur] = true;
            loop.Add(start);
            loop.Add(target);

            while (true)
            {
                if (!map.TryGetValue(target, out var lst)) break;
                int next = -1;
                foreach (var idx in lst)
                {
                    if (used[idx]) continue;
                    var s = segs[idx];
                    if (Approximately(s.a, target, snapEps)) { next = idx; target = s.b; break; }
                    if (Approximately(s.b, target, snapEps)) { next = idx; target = s.a; break; }
                }
                if (next < 0) break;
                used[next] = true;
                loop.Add(target);
                if (Approximately(loop[0], target, snapEps)) break;
            }
            if (loop.Count >= 3) loops.Add(loop);
        }
        return loops;

        void Add(Dictionary<Vector3, List<int>> m, Vector3 k, int idx)
        { if (!m.TryGetValue(k, out var l)) { l = new List<int>(); m[k] = l; } l.Add(idx); }
        bool Approximately(Vector3 a, Vector3 b, float eps) => (a - b).sqrMagnitude <= eps * eps;
    }

    static float SignedArea2D(List<Vector3> poly)
    {
        double s = 0.0; int n = poly.Count;
        for (int i = 0; i < n; i++)
        {
            var p = poly[i]; var q = poly[(i + 1) % n];
            s += (double)p.x * q.y - (double)q.x * p.y;
        }
        return (float)(0.5 * s);
    }

    static void ClassifyLoops(List<Seg> segs, out List<Seg> outerSegs, out List<Seg> holeSegs)
    {
        outerSegs = new List<Seg>(); holeSegs = new List<Seg>();
        if (segs.Count == 0) return;

        var loops = BuildLoopsFromSegments(segs);
        if (loops.Count == 0) { outerSegs.AddRange(segs); return; }

        int outerIdx = 0; float maxAbs = 0f;
        for (int i = 0; i < loops.Count; i++)
        {
            float a = Mathf.Abs(SignedArea2D(loops[i]));
            if (a > maxAbs) { maxAbs = a; outerIdx = i; }
        }

        var clsMap = new Dictionary<(Vector3, Vector3), BoundaryClass>(new SegKeyEq(1e-4f));
        for (int i = 0; i < loops.Count; i++)
        {
            var cls = (i == outerIdx) ? BoundaryClass.Outer : BoundaryClass.Hole;
            var loop = loops[i];
            for (int k = 0; k < loop.Count; k++)
            {
                var a = loop[k]; var b = loop[(k + 1) % loop.Count];
                clsMap[(a, b)] = cls; clsMap[(b, a)] = cls;
            }
        }

        foreach (var s in segs)
        {
            var key = (s.a, s.b);
            if (clsMap.TryGetValue(key, out var c))
            {
                var ss = new Seg(s.a, s.b, c);
                if (c == BoundaryClass.Outer) outerSegs.Add(ss); else holeSegs.Add(ss);
            }
            else
            {
                var ss = new Seg(s.a, s.b, BoundaryClass.Outer);
                outerSegs.Add(ss);
            }
        }
    }

    class Vec3KeyEq : IEqualityComparer<Vector3>
    {
        readonly float eps;
        public Vec3KeyEq(float e) { eps = e; }
        public bool Equals(Vector3 a, Vector3 b) => (a - b).sqrMagnitude <= eps * eps;
        public int GetHashCode(Vector3 v)
        {
            unchecked
            {
                return Mathf.RoundToInt(v.x / eps) * 73856093
                     ^ Mathf.RoundToInt(v.y / eps) * 19349663
                     ^ Mathf.RoundToInt(v.z / eps) * 83492791;
            }
        }
    }
    class SegKeyEq : IEqualityComparer<(Vector3, Vector3)>
    {
        readonly float eps;
        Vec3KeyEq veq;
        public SegKeyEq(float e) { eps = e; veq = new Vec3KeyEq(e); }
        public bool Equals((Vector3, Vector3) x, (Vector3, Vector3) y)
            => veq.Equals(x.Item1, y.Item1) && veq.Equals(x.Item2, y.Item2);
        public int GetHashCode((Vector3, Vector3) k)
        {
            unchecked { return veq.GetHashCode(k.Item1) * 486187739 ^ veq.GetHashCode(k.Item2); }
        }
    }

    static void SampleSegmentsHoleAware(List<Seg> segsOuter, List<Seg> segsHole,
        int totalEdgeCount, float holeWeight,
        List<Vector3> outPts, List<BoundaryClass> outCls)
    {
        outPts.Capacity += totalEdgeCount;
        outCls.Capacity += totalEdgeCount;

        float Len(List<Seg> s)
        {
            float sum = 0f; for (int i = 0; i < s.Count; i++) sum += (s[i].a - s[i].b).magnitude; return sum;
        }
        float Louter = Len(segsOuter);
        float Lhole = Len(segsHole);

        float wOuter = Louter;
        float wHole = Lhole * Mathf.Max(1f, holeWeight);
        float wSum = Mathf.Max(1e-6f, wOuter + wHole);

        int nHole = Mathf.RoundToInt(totalEdgeCount * (wHole / wSum));
        nHole = Mathf.Clamp(nHole, 0, totalEdgeCount);
        int nOuter = totalEdgeCount - nHole;

        void Sample(List<Seg> S, int count, BoundaryClass cls)
        {
            if (count <= 0 || S.Count == 0) return;
            var lens = new float[S.Count];
            float tot = 0f;
            for (int i = 0; i < S.Count; i++) { lens[i] = (S[i].a - S[i].b).magnitude; tot += lens[i]; }
            for (int k = 0; k < count; k++)
            {
                float r = UnityEngine.Random.value * tot;
                int idx = 0; float acc = 0f;
                for (; idx < S.Count; idx++) { acc += lens[idx]; if (r <= acc) break; }
                idx = Mathf.Clamp(idx, 0, S.Count - 1);
                float t = UnityEngine.Random.value;
                var p = Vector3.Lerp(S[idx].a, S[idx].b, t);
                outPts.Add(p);
                outCls.Add(cls);
            }
        }

        Sample(segsOuter, nOuter, BoundaryClass.Outer);
        Sample(segsHole, nHole, BoundaryClass.Hole);
    }




    void ComputeHoleWHForObject(GameObject go, bool poseInv,
      out Vector2 outerWH, out List<Vector2> holeWH, out List<float> holeSizeScore)
    {
        outerWH = Vector2.zero;
        holeWH = new List<Vector2>();
        holeSizeScore = new List<float>(); // W*H (normalize edilmemiş) — sıralama ağırlığı

        // 1) Boundary segmentlerini topla
        var segsRaw = new List<Seg>();
        foreach (var f in go.GetComponentsInChildren<MeshFilter>(true))
            if (f.sharedMesh) CollectBoundarySegments(f.sharedMesh, f.transform, go.transform, poseInv, segsRaw);
        if (segsRaw.Count == 0) return;

        // 2) Outer / hole ayır
        List<Seg> segsOuter, segsHole;
        ClassifyLoops(segsRaw, out segsOuter, out segsHole);

        // 3) Loop’ları kur
        var loopsOuter = BuildLoopsFromSegments(segsOuter);
        var loopsHole = BuildLoopsFromSegments(segsHole);

        // 4) En büyük outer loop → outerWH (AABB)
        int bestIdx = -1; float bestAbs = 0f;
        for (int i = 0; i < loopsOuter.Count; i++)
        {
            float a = Mathf.Abs(SignedArea2D(loopsOuter[i]));
            if (a > bestAbs) { bestAbs = a; bestIdx = i; }
        }
        if (bestIdx >= 0)
        {
            Bounds ob = BoundsFromPoints(loopsOuter[bestIdx]);
            outerWH = new Vector2(Mathf.Max(1e-9f, ob.size.x), Mathf.Max(1e-9f, ob.size.y));
        }

        // 5) Delikler → W/H listesi (gürültüyü ele)
        for (int i = 0; i < loopsHole.Count; i++)
        {
            var L = loopsHole[i];
            if (L == null || L.Count < 3) continue;

            float areaAbs = Mathf.Abs(SignedArea2D(L));
            if (areaAbs < 1e-8f) continue;

            Bounds hb = BoundsFromPoints(L);
            float w = hb.size.x;
            float h = hb.size.y;
            if (w <= 1e-6f || h <= 1e-6f) continue;

            holeWH.Add(new Vector2(w, h));
            holeSizeScore.Add(w * h);
        }

        // 6) Büyüklüğe göre (W*H) büyükten küçüğe sırala
        //    (LINQ/lambda kullanmadan, index listesiyle)
        int n = holeWH.Count;
        if (n <= 1) return;

        // indeks dizisi
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;

        // selection sort (büyükten küçüğe)
        for (int i = 0; i < n - 1; i++)
        {
            int maxJ = i;
            float maxVal = holeSizeScore[idx[i]];
            for (int j = i + 1; j < n; j++)
            {
                float v = holeSizeScore[idx[j]];
                if (v > maxVal) { maxVal = v; maxJ = j; }
            }
            if (maxJ != i)
            {
                int tmp = idx[i];
                idx[i] = idx[maxJ];
                idx[maxJ] = tmp;
            }
        }

        // yeniden sıralı kopyalar
        var holeWH2 = new List<Vector2>(n);
        var holeScore2 = new List<float>(n);
        for (int k = 0; k < n; k++)
        {
            int ii = idx[k];
            holeWH2.Add(holeWH[ii]);
            holeScore2.Add(holeSizeScore[ii]);
        }
        holeWH = holeWH2;
        holeSizeScore = holeScore2;
    }





    float ComputeHoleWHCountPenalty01(
    GameObject objA, GameObject objB, bool poseInv,
    float wWHEach, float wCount)
    {
        // 1) W/H listeleri + outerWH
        ComputeHoleWHForObject(objA, poseInv, out var outerA, out var WHA, out var scoreA);
        ComputeHoleWHForObject(objB, poseInv, out var outerB, out var WHB, out var scoreB);

        // Delik yoksa ceza 0
        if ((WHA.Count == 0 && WHB.Count == 0)) return 0f;

        // 2) Normalize (ölçekten bağımsız)
        Vector2 nA = new Vector2(Mathf.Max(1e-9f, outerA.x), Mathf.Max(1e-9f, outerA.y));
        Vector2 nB = new Vector2(Mathf.Max(1e-9f, outerB.x), Mathf.Max(1e-9f, outerB.y));

        var NWA = new List<Vector2>(WHA.Count);
        var NWB = new List<Vector2>(WHB.Count);
        var SA = new List<float>(scoreA.Count);
        var SB = new List<float>(scoreB.Count);

        for (int i = 0; i < WHA.Count; i++)
        {
            var v = WHA[i];
            NWA.Add(new Vector2(v.x / nA.x, v.y / nA.y));
            SA.Add((v.x / nA.x) * (v.y / nA.y)); // normalize büyüklük ~ alan proxy
        }
        for (int i = 0; i < WHB.Count; i++)
        {
            var v = WHB[i];
            NWB.Add(new Vector2(v.x / nB.x, v.y / nB.y));
            SB.Add((v.x / nB.x) * (v.y / nB.y));
        }

        // 3) Listeleri eşit uzunluğa getir (eksikler 0’larla doldurulur)
        int m = Mathf.Max(NWA.Count, NWB.Count);
        while (NWA.Count < m) { NWA.Add(Vector2.zero); SA.Add(0f); }
        while (NWB.Count < m) { NWB.Add(Vector2.zero); SB.Add(0f); }

        // 4) Top‑k (büyüklüğe göre) ve p‑norm fark
        int keep = Mathf.Max(1, Mathf.RoundToInt(m * Mathf.Clamp01(holeWHTopFrac)));
        double num = 0.0, den = 0.0;
        float p = Mathf.Max(1f, holeWHExp);

        for (int i = 0; i < keep; i++)
        {
            // her iki taraftaki i’in büyüklük proxysi
            float s = Mathf.Max(SA[i], SB[i]);      // büyük olana ağırlık
            if (holeWHRankWeighted)
            {
                float rankW = 1f - (i / (float)keep);
                s *= Mathf.Clamp(rankW, 0.001f, 1f);
            }

            // ΔW, ΔH (relative)
            float aW = NWA[i].x, aH = NWA[i].y;
            float bW = NWB[i].x, bH = NWB[i].y;

            float dW = (Mathf.Abs(aW - bW)) / Mathf.Max(1e-6f, Mathf.Max(aW, bW));
            float dH = (Mathf.Abs(aH - bH)) / Mathf.Max(1e-6f, Mathf.Max(aH, bH));

            // p-norm karışımı
            float mix = Mathf.Pow(0.5f * (Mathf.Pow(dW, p) + Mathf.Pow(dH, p)), 1f / p);
            num += s * mix; den += s;
        }

        float whPun = (den > 1e-9) ? Mathf.Clamp01((float)(num / den)) : 0f;

        // 5) Count cezası (sadece sayıya bakar)
        float countPun = 0f;
        if (useHoleCountOnly)
        {
            int cA = WHA.Count, cB = WHB.Count;
            int cMax = Mathf.Max(cA, cB);
            if (cMax > 0)
            {
                float r = Mathf.Abs(cA - cB) / (float)cMax; // 0..1
                countPun = r;
            }
        }

        // 6) Ağırlıkla birleştir ve döndür (ceza 0..1)
        float wSum = Mathf.Max(1e-6f, wWHEach + wCount);
        float penalty = (wWHEach / wSum) * whPun + (wCount / wSum) * countPun;
        penalty = Mathf.Clamp01(holeWHPenaltyLambda * penalty);  // genel şiddet
        return Mathf.Clamp01(penalty);
    }






















    // --- MAIN ---
    [ContextMenu("Compute Now")]
    public void ComputeAll()
    {
        if (objectA == null || objectB == null)
        {
            UnityEngine.Debug.LogWarning("[MeshSimilarity] Lütfen Object A ve Object B atayın.");
            return;
        }

        var swTotal = Stopwatch.StartNew();
        var rngState = UnityEngine.Random.state;

        // ---------- 1) ÖRNEKLEME ----------
        ptsA.Clear(); ptsB.Clear();
        clsA.Clear(); clsB.Clear();

        if (emphasizeEdges)
        {
            int nEdge = Mathf.RoundToInt(samplesPerMesh * Mathf.Clamp01(edgePortion));
            int nSurf = Mathf.Max(0, samplesPerMesh - nEdge);

            var segsA_raw = new List<Seg>(); var segsB_raw = new List<Seg>();
            foreach (var f in objectA.GetComponentsInChildren<MeshFilter>(true))
                if (f.sharedMesh) CollectBoundarySegments(f.sharedMesh, f.transform, objectA.transform, poseInvariant, segsA_raw);
            foreach (var f in objectB.GetComponentsInChildren<MeshFilter>(true))
                if (f.sharedMesh) CollectBoundarySegments(f.sharedMesh, f.transform, objectB.transform, poseInvariant, segsB_raw);

            // outer/hole sınıflandır
            List<Seg> segsA_outer, segsA_hole, segsB_outer, segsB_hole;
            ClassifyLoops(segsA_raw, out segsA_outer, out segsA_hole);
            ClassifyLoops(segsB_raw, out segsB_outer, out segsB_hole);

            // A için
            UnityEngine.Random.InitState(randomSeed ^ 0x34567);
            if (holeAwareMatching)
                SampleSegmentsHoleAware(segsA_outer, segsA_hole, nEdge, holeEdgeWeight, ptsA, clsA);
            else
            {
                SampleSegments(segsA_raw, nEdge, ptsA);
                for (int i = 0; i < nEdge; i++) clsA.Add(BoundaryClass.Unknown);
            }

            // B için
            UnityEngine.Random.InitState(randomSeed ^ 0x89AB);
            if (holeAwareMatching)
                SampleSegmentsHoleAware(segsB_outer, segsB_hole, nEdge, holeEdgeWeight, ptsB, clsB);
            else
            {
                SampleSegments(segsB_raw, nEdge, ptsB);
                for (int i = 0; i < nEdge; i++) clsB.Add(BoundaryClass.Unknown);
            }

            // yüzey örneklemesi (boundary-only değilse)
            if (!hausdorffBoundaryOnly)
            {
                int beforeA = ptsA.Count;
                SampleFromObjectDeterministic(objectA, objectA.transform, nSurf, poseInvariant, randomSeed ^ 1111, ptsA);
                for (int i = 0; i < (ptsA.Count - beforeA); i++) clsA.Add(BoundaryClass.Unknown);

                int beforeB = ptsB.Count;
                SampleFromObjectDeterministic(objectB, objectB.transform, nSurf, poseInvariant, randomSeed ^ 2222, ptsB);
                for (int i = 0; i < (ptsB.Count - beforeB); i++) clsB.Add(BoundaryClass.Unknown);
            }
        }
        else
        {
            // saf yüzey örnekleme
            SampleFromObjectDeterministic(objectA, objectA.transform, samplesPerMesh, poseInvariant, randomSeed, ptsA);
            for (int i = 0; i < ptsA.Count; i++) clsA.Add(BoundaryClass.Unknown);

            SampleFromObjectDeterministic(objectB, objectB.transform, samplesPerMesh, poseInvariant, randomSeed, ptsB);
            for (int i = 0; i < ptsB.Count; i++) clsB.Add(BoundaryClass.Unknown);
        }

        // ---------- 2) HİZALAMA / PLANAR PROJEKSİYON ----------
        if (!perChildMode && centerAlignByBounds) CenterAlignByBoundsPts(ptsA, ptsB);
        if (!perChildMode && (forcePlanarFor2D || projectPlanar)) ProjectToBestPlane(ptsA, ptsB);

        // ---------- 3) NORMALİZE DİAGONAL ----------
        if (poseInvariant)
        {
            var bA = BoundsFromPoints(ptsA);
            var bB = BoundsFromPoints(ptsB);
            Vector3 min = Vector3.Min(bA.min, bB.min);
            Vector3 max = Vector3.Max(bA.max, bB.max);
            bboxDiagonal = (max - min).magnitude;
        }
        else
        {
            bboxDiagonal = CombinedWorldDiag(objectA, objectB);
        }
        if (bboxDiagonal <= 1e-8f) bboxDiagonal = 1e-8f;

        // ---------- 4) CHAMFER / HAUSDORFF ----------
        var swChamfer = Stopwatch.StartNew();
        nnLinesA2B.Clear(); nnLinesB2A.Clear();
        matchRecall = matchPrecision = matchF1 = 0f;
        recallStr = "-"; f1Str = "-";

        if (perChildMode)
        {
            modeStr = "Per-Child";
            float worstChild;
            float meanChildSim = ChildwiseSimilarity01(objectA, objectB, out worstChild);

            geometrySimilarity01 = Mathf.Pow(Mathf.Clamp01(meanChildSim), Mathf.Max(1f, simSharpnessPow));
            float normValChild = Mathf.Clamp01(1f - geometrySimilarity01);
            chamferDistance = normValChild * bboxDiagonal; // log amaçlı
        }
        else
        {
            modeStr = "Global";
            float recallAB = 0f, recallBA = 0f;
            float chAB, chBA;

            if (metric == SimilarityMetric.TrimmedHausdorff)
            {
                if (holeAwareMatching)
                {
                    chAB = PercentileNearestDistHoleAware(
     ptsA, clsA, ptsB, clsB, bboxDiagonal,
     useCapped ? capAtDiagFrac : 0f, out recallAB, matchDeltaFrac, hausdorffPercentile,
     holePenaltyMultiplier,
     drawNearestNeighborLines ? nnLinesA2B : null, maxNNLinesPerSet);

                    chBA = PercentileNearestDistHoleAware(
                        ptsB, clsB, ptsA, clsA, bboxDiagonal,
                        useCapped ? capAtDiagFrac : 0f, out recallBA, matchDeltaFrac, hausdorffPercentile,
                        holePenaltyMultiplier,
                        drawNearestNeighborLines ? nnLinesB2A : null, maxNNLinesPerSet);

                }
                else
                {
                    chAB = PercentileNearestDist(
                        ptsA, ptsB, bboxDiagonal,
                        useCapped ? capAtDiagFrac : 0f, out recallAB, matchDeltaFrac, hausdorffPercentile);

                    chBA = PercentileNearestDist(
                        ptsB, ptsA, bboxDiagonal,
                        useCapped ? capAtDiagFrac : 0f, out recallBA, matchDeltaFrac, hausdorffPercentile);
                }
            }
            else // ChamferMean
            {
                chAB = AvgNearestDistRobust(
                    ptsA, ptsB, bboxDiagonal,
                    useTrimmed ? trimTopPercent : 0f,
                    useCapped ? capAtDiagFrac : 0f,
                    drawNearestNeighborLines ? nnLinesA2B : null, maxNNLinesPerSet,
                    out recallAB, matchDeltaFrac);

                chBA = AvgNearestDistRobust(
                    ptsB, ptsA, bboxDiagonal,
                    useTrimmed ? trimTopPercent : 0f,
                    useCapped ? capAtDiagFrac : 0f,
                    drawNearestNeighborLines ? nnLinesB2A : null, maxNNLinesPerSet,
                    out recallBA, matchDeltaFrac);
            }

            // Hausdorff yön birleştirme
            if (metric == SimilarityMetric.TrimmedHausdorff)
                chamferDistance = (hausdorffAggregation == HausdorffAgg.MaxOfDirs) ? Mathf.Max(chAB, chBA) : 0.5f * (chAB + chBA);
            else
                chamferDistance = 0.5f * (chAB + chBA);

            // ---------- 4.1) ÖLÇEK CEZASI (opsiyonel) ----------
            if (useScalePenalty)
            {
                var bA = BoundsFromPoints(ptsA);
                var bB = BoundsFromPoints(ptsB);

                Vector3 sA = bA.size;
                Vector3 sB = bB.size;

                float dx = Mathf.Abs(sA.x - sB.x) / Mathf.Max(1e-6f, Mathf.Max(sA.x, sB.x));
                float dy = Mathf.Abs(sA.y - sB.y) / Mathf.Max(1e-6f, Mathf.Max(sA.y, sB.y));
                float dz = Mathf.Abs(sA.z - sB.z) / Mathf.Max(1e-6f, Mathf.Max(sA.z, sB.z));

                float wsumAxes = Mathf.Max(1e-6f, axisPenaltyWeights.x + axisPenaltyWeights.y + axisPenaltyWeights.z);
                float relSizeDiff = (axisPenaltyWeights.x * dx + axisPenaltyWeights.y * dy + axisPenaltyWeights.z * dz) / wsumAxes;

                float extraScale = scalePenaltyLambda * relSizeDiff * bboxDiagonal;
                chamferDistance += extraScale;
            }

            // ---------- 4.2) DELİK CEZASI (0..1) → BENZERLİKTE KARIŞTIR ----------
            // Not: Delik etkisini artık mesafeye eklemiyoruz; benzerlikte ağırlıklı karıştırıyoruz.
            // ---------- Outer Sim ----------
            float outerSim01;
            {
                float normValOuter = Mathf.Clamp01(chamferDistance / bboxDiagonal);
                outerSim01 = 1f - normValOuter;
            }

            if (holeAwareMatching && (useHoleWHPenalty || useHoleCountOnly))
            {
                float wWH = useHoleWHPenalty ? 1f : 0f;
                float wCnt = useHoleCountOnly ? holeCountPenaltyLambda : 0f;

                float holePenalty01 = ComputeHoleWHCountPenalty01(
                    objectA, objectB, poseInvariant,
                    wWH, wCnt);

                float holeSim01 = 1f - holePenalty01;

                // Ağırlıklar
                float outerWeight = 0.30f;
                float holeWeight = 0.70f;
                float wsum = Mathf.Max(1e-6f, outerWeight + holeWeight);
                outerWeight /= wsum; holeWeight /= wsum;

                float combinedSim01 = Mathf.Clamp01(outerWeight * outerSim01 + holeWeight * holeSim01);
                geometrySimilarity01 = Mathf.Pow(combinedSim01, Mathf.Max(1f, simSharpnessPow));
            }
            else
            {
                geometrySimilarity01 = Mathf.Pow(Mathf.Clamp01(outerSim01), Mathf.Max(1f, simSharpnessPow));
            }

            // recall / precision / F1 (yaklaşık)
            matchRecall = 0.5f * (recallAB + recallBA);
            matchPrecision = matchRecall;
            matchF1 = (matchPrecision + matchRecall > 1e-6f)
                ? (2f * matchPrecision * matchRecall) / (matchPrecision + matchRecall)
                : 0f;

            // Kısa log stringleri
            recallStr = $"{matchRecall:P1}";
            f1Str = (matchF1 > 0f) ? $"{matchF1:P1}" : "-";
        }

        swChamfer.Stop();
        swTotal.Stop();

        UnityEngine.Debug.Log(
            $"[MeshSimilarity] Mode: {modeStr}\n" +
            $"- Samples per mesh: {samplesPerMesh}\n" +
            $"- A points: {ptsA.Count}, B points: {ptsB.Count}\n" +
            $"- Metric: {metric}, Value: {chamferDistance:F6}\n" +
            $"- Diagonal: {bboxDiagonal:F6}\n" +
            $"- Similarity (0..1): {geometrySimilarity01:F6}\n" +
            $"- Recall(A→B,B→A,avg): {recallStr}\n" +
            $"- F1 (≈match quality): {f1Str}\n" +
            $"- Chamfer time: {swChamfer.ElapsedMilliseconds} ms, Total: {swTotal.ElapsedMilliseconds} ms");

        if (Application.isPlaying && spawnPointMarkers && pointPrefab != null)
            SpawnMarkers();

        UnityEngine.Random.state = rngState;
#if UNITY_EDITOR
        UnityEditor.SceneView.RepaintAll();
#endif
    }



    // Hole-aware percentile NN distance (A→B)
    float PercentileNearestDistHoleAware(
     List<Vector3> A, List<BoundaryClass> clsA,
     List<Vector3> B, List<BoundaryClass> clsB,
     float diag, float capFrac, out float recall,
     float matchDeltaFrac, float percentile,
     float mismatchPenaltyMul,
     List<NNLine> debugLines = null, int maxLines = 0)
    {
        recall = 0f;
        if (A == null || B == null || A.Count == 0 || B.Count == 0)
            return 0f;

        float cap = capFrac * diag;
        float matchDelta = matchDeltaFrac * diag;

        List<float> distsPow = new List<float>(A.Count);
        int hits = 0;
        int lineCounter = 0;

        // Boundary class index listeleri
        var idxOuter = new List<int>();
        var idxHole = new List<int>();
        for (int i = 0; i < clsB.Count; i++)
        {
            if (clsB[i] == BoundaryClass.Outer) idxOuter.Add(i);
            else if (clsB[i] == BoundaryClass.Hole) idxHole.Add(i);
        }

        for (int i = 0; i < A.Count; i++)
        {
            Vector3 p = A[i];
            var ca = (i < clsA.Count) ? clsA[i] : BoundaryClass.Unknown;

            List<int> tgt = null;
            if (ca == BoundaryClass.Hole) tgt = idxHole;
            else if (ca == BoundaryClass.Outer) tgt = idxOuter;

            float best = float.PositiveInfinity;
            int bestIdx = -1;

            if (tgt != null && tgt.Count > 0)
            {
                for (int ti = 0; ti < tgt.Count; ti++)
                {
                    int j = tgt[ti];
                    float dsq = (p - B[j]).sqrMagnitude;
                    if (dsq < best) { best = dsq; bestIdx = j; }
                }
                best = Mathf.Sqrt(best);
            }
            else
            {
                for (int j = 0; j < B.Count; j++)
                {
                    float dsq = (p - B[j]).sqrMagnitude;
                    if (dsq < best) { best = dsq; bestIdx = j; }
                }
                best = Mathf.Sqrt(best) * Mathf.Max(1f, mismatchPenaltyMul);
            }

            if (best > cap) best = cap;

            // Debug çizgisi ekleme
            if (debugLines != null && lineCounter < maxLines && bestIdx >= 0)
            {
                debugLines.Add(new NNLine(p, B[bestIdx]));
                lineCounter++;
            }

            distsPow.Add(Mathf.Pow(best, distanceExponent));
            if (best <= matchDelta) hits++;
        }

        distsPow.Sort();
        int idx = Mathf.Clamp(Mathf.RoundToInt(percentile * (distsPow.Count - 1)), 0, distsPow.Count - 1);
        float val = Mathf.Pow(distsPow[idx], 1f / distanceExponent);

        recall = (float)hits / Mathf.Max(1, A.Count);
        return val;
    }



    // Delik/outer döngülerinden alan ve çevre istatistiklerini üretir.
    // Not: 2D düzlem varsayımı (projectPlanar/forcePlanarFor2D sonrası noktalar XY'dedir).
    void ComputeHoleStatsForObject(GameObject go, bool poseInv,
        out float outerArea, out float outerPerim,
        out List<float> holeAreas, out List<float> holePerims)
    {
        outerArea = 0f; outerPerim = 0f;
        holeAreas = new List<float>(); holePerims = new List<float>();

        // 1) Tüm boundary segmentlerini topla
        var segsRaw = new List<Seg>();
        foreach (var f in go.GetComponentsInChildren<MeshFilter>(true))
            if (f.sharedMesh) CollectBoundarySegments(f.sharedMesh, f.transform, go.transform, poseInv, segsRaw);

        if (segsRaw.Count == 0)
            return;

        // 2) Loop'ları kur ve outer/hole sınıfı ver
        List<Seg> segsOuter, segsHole;
        ClassifyLoops(segsRaw, out segsOuter, out segsHole);

        // 3) Loop'ları yeniden kurmak için outer ve hole segmentlerini ayrı ayrı topla
        List<List<Vector3>> loopsOuter = BuildLoopsFromSegments(segsOuter);
        List<List<Vector3>> loopsHole = BuildLoopsFromSegments(segsHole);

        // 4) Outer: en büyük alanlı loop'u outer sayalım (ClassifyLoops zaten böyle işaretliyor ama
        //    loop parçalanmış ise buradan ek güvence sağlarız)
        float maxAbsArea = 0f;
        int outerIdx = -1;
        for (int i = 0; i < loopsOuter.Count; i++)
        {
            float a = Mathf.Abs(SignedArea2D(loopsOuter[i]));
            if (a > maxAbsArea) { maxAbsArea = a; outerIdx = i; }
        }
        if (outerIdx >= 0)
        {
            outerArea = Mathf.Abs(SignedArea2D(loopsOuter[outerIdx]));
            // çevre:
            float per = 0f;
            var L = loopsOuter[outerIdx];
            for (int k = 0; k < L.Count; k++)
                per += Vector3.Distance(L[k], L[(k + 1) % L.Count]);
            outerPerim = Mathf.Max(outerPerim, per);
        }

        // 5) Delikler: her loop'un alan/çevresini al
        for (int i = 0; i < loopsHole.Count; i++)
        {
            var L = loopsHole[i];
            if (L.Count < 3) continue;
            float a = Mathf.Abs(SignedArea2D(L));
            float p = 0f;
            for (int k = 0; k < L.Count; k++)
                p += Vector3.Distance(L[k], L[(k + 1) % L.Count]);

            // gürültü deliklerini ele
            if (a > 1e-8f && p > 1e-6f)
            {
                holeAreas.Add(a);
                holePerims.Add(p);
            }
        }

        // 6) Büyükten küçüğe sırala (eşleştirirken en büyüğü en büyüğe eşleyelim)
        holeAreas.Sort((x, y) => y.CompareTo(x));
        holePerims.Sort((x, y) => y.CompareTo(x));
    }


    // ---------- Parça-bazlı benzerlik ----------
    float ChildwiseSimilarity01(GameObject A, GameObject B, out float worstChildSim01)
    {
        var fa = A.GetComponentsInChildren<MeshFilter>(true);
        var fb = B.GetComponentsInChildren<MeshFilter>(true);

        // Hiyerarşi yoluna göre sırala (eşleşme kararlı olsun)
        Array.Sort(fa, (x, y) => string.CompareOrdinal(GetHierarchyPath(x.transform), GetHierarchyPath(y.transform)));
        Array.Sort(fb, (x, y) => string.CompareOrdinal(GetHierarchyPath(x.transform), GetHierarchyPath(y.transform)));

        int m = Mathf.Min(fa.Length, fb.Length);
        if (m == 0) { worstChildSim01 = 0f; return 0f; }

        // Çocuk başına örnek sayısı (üstteki samplesPerMesh toplamdan pay edilir)
        int perChild = Mathf.Max(200, samplesPerMesh / Mathf.Max(1, m));

        List<float> childSims = new List<float>(m);
        double sumSim = 0.0;
        float worstSim = 1f;

        for (int i = 0; i < m; i++)
        {
            var ma = fa[i].sharedMesh;
            var mb = fb[i].sharedMesh;

            // Rotasyon bazlı karşılaştırma istendiği için, geometri örnekleme opsiyonel.
            // Yine de shapeSim istenirse üretilebilsin diye örnekleyip hazır tutacağız.
            float shapeSim = 1f; // sade: default 1 (rotation-only modunda kullanılmaz)
            if ((rotMode != RotationPenaltyMode.RotationOnly) && ma != null && mb != null)
            {
                var pa = new List<Vector3>(perChild);
                var pb = new List<Vector3>(perChild);
                var na = new List<Vector3>(perChild);
                var nb = new List<Vector3>(perChild);

                int seedA = CombineSeed(randomSeed, ma.GetInstanceID(), fa[i].transform.GetInstanceID());
                int seedB = CombineSeed(randomSeed, mb.GetInstanceID(), fb[i].transform.GetInstanceID());
                var prev = UnityEngine.Random.state;

                UnityEngine.Random.InitState(seedA);
                SampleSurfacePointsWithNormals(ma, fa[i].transform, A.transform, perChild, poseInvariant, pa, na);

                UnityEngine.Random.InitState(seedB);
                SampleSurfacePointsWithNormals(mb, fb[i].transform, B.transform, perChild, poseInvariant, pb, nb);

                UnityEngine.Random.state = prev;

                if (childCenterAlignByBounds)
                {
                    var bA0 = BoundsFromPoints(pa);
                    var bB0 = BoundsFromPoints(pb);
                    Vector3 ca = bA0.center, cb = bB0.center;
                    for (int k = 0; k < pa.Count; k++) pa[k] -= ca;
                    for (int k = 0; k < pb.Count; k++) pb[k] -= cb;
                }

                var bA2 = BoundsFromPoints(pa);
                var bB2 = BoundsFromPoints(pb);
                float d = (Vector3.Max(bA2.max, bB2.max) - Vector3.Min(bA2.min, bB2.min)).magnitude;
                if (d <= 1e-8f) d = 1e-8f;

                float chAB, chBA;
                if (useNormalPenalty && normalPenaltyLambda > 0f)
                {
                    chAB = ChamferWithNormalPenalty(pa, na, pb, nb, d, normalPenaltyLambda);
                    chBA = ChamferWithNormalPenalty(pb, nb, pa, na, d, normalPenaltyLambda);
                }
                else
                {
                    chAB = AvgNearestDistRobust(
                        pa, pb, 1f,
                        useTrimmed ? trimTopPercent : 0f,
                        useCapped ? capAtDiagFrac : 0f,
                        null, 0, out _, matchDeltaFrac);

                    chBA = AvgNearestDistRobust(
                        pb, pa, 1f,
                        useTrimmed ? trimTopPercent : 0f,
                        useCapped ? capAtDiagFrac : 0f,
                        null, 0, out _, matchDeltaFrac);
                }

                float ch = 0.5f * (chAB + chBA);
                shapeSim = 1f - Mathf.Clamp01(ch / d);
            }

            // Rotasyon referansı
            Quaternion qA = GetRefRotation(fa[i].transform, A.transform);
            Quaternion qB = GetRefRotation(fb[i].transform, B.transform);
            float thetaDeg = Quaternion.Angle(qA, qB);
            float rotSim = useRotationPenalty ? RotationSimilarityFromTheta(thetaDeg) : 1f;

            // Pozisyon referansı ve benzerlik
            float posSim = 1f;
            if (usePositionPenalty)
            {
                Vector3 pA = GetRefPosition(fa[i].transform, A.transform);
                Vector3 pB = GetRefPosition(fb[i].transform, B.transform);
                float delta = Vector3.Distance(pA, pB);
                posSim = PositionSimilarityFromDelta(delta);
            }

            // rotSim + posSim → poseSim
            float poseSim;
            switch (poseCombine)
            {
                case PoseCombine.Multiplicative: poseSim = rotSim * posSim; break;
                case PoseCombine.LinearBlend: poseSim = Mathf.Lerp(rotSim, posSim, Mathf.Clamp01(poseBlendWeight)); break;
                case PoseCombine.Min:
                default: poseSim = Mathf.Min(rotSim, posSim); break;
            }

            // LOG
            UnityEngine.Debug.Log($"[PerChild Pose] idx={i} A={fa[i].name} B={fb[i].name} θ={thetaDeg:F2}° rotSim={rotSim:F3} posSim={posSim:F3} -> poseSim={poseSim:F3} shapeSim={shapeSim:F3}");

            // finalSim: rotMode'a göre şekil ile birleştir
            float finalSim;
            if (rotMode == RotationPenaltyMode.RotationOnly)
                finalSim = poseSim; // sadece pose (rot+pos)
            else if (rotMode == RotationPenaltyMode.Multiplicative)
                finalSim = shapeSim * poseSim;
            else if (rotMode == RotationPenaltyMode.LinearBlend)
                finalSim = Mathf.Lerp(shapeSim, poseSim, Mathf.Clamp01(rotWeight));
            else // MaxOfBoth
                finalSim = Mathf.Min(shapeSim, poseSim);

            childSims.Add(finalSim);
            sumSim += finalSim;
            if (finalSim < worstSim) worstSim = finalSim;
        }

        worstChildSim01 = worstSim;

        switch (childAggregation)
        {
            case ChildAgg.Mean:
                return (float)(sumSim / childSims.Count);

            case ChildAgg.GeometricMean:
                {
                    double prod = 1.0;
                    for (int i = 0; i < childSims.Count; i++)
                        prod *= Mathf.Clamp01(childSims[i]);
                    return (float)Math.Pow(prod, 1.0 / Math.Max(1, childSims.Count));
                }

            case ChildAgg.Min:
                return worstSim;

            case ChildAgg.Pct25:
                {
                    childSims.Sort();
                    int k = Mathf.Clamp(Mathf.RoundToInt(0.25f * (childSims.Count - 1)), 0, childSims.Count - 1);
                    return childSims[k];
                }
        }
        return (float)(sumSim / childSims.Count);
    }

    Quaternion GetRefRotation(Transform child, Transform root)
    {
        switch (rotationReference)
        {
            case RotationReference.World:
                return child.rotation;
            case RotationReference.Local:
                return child.localRotation;
            case RotationReference.RootRelative:
            default:
                return Quaternion.Inverse(root.rotation) * child.rotation;
        }
    }

    void SampleSurfacePointsWithNormals(
        Mesh mesh, Transform tr, Transform root, int count, bool poseInv,
        List<Vector3> outPts, List<Vector3> outNrms)
    {
        var verts = mesh.vertices;
        var tris = mesh.triangles;
        var norms = mesh.normals;
        int triCount = tris.Length / 3;
        if (triCount == 0 || verts == null || verts.Length == 0) return;

        var triAreas = new float[triCount];
        float totalArea = 0f;
        for (int i = 0; i < triCount; i++)
        {
            var v0 = verts[tris[3 * i]];
            var v1 = verts[tris[3 * i + 1]];
            var v2 = verts[tris[3 * i + 2]];
            float area = Vector3.Cross(v1 - v0, v2 - v0).magnitude * 0.5f;
            triAreas[i] = area; totalArea += area;
        }

        if (totalArea <= 1e-10f)
        {
            int n = Mathf.Min(count, verts.Length);
            for (int i = 0; i < n; i++)
            {
                Vector3 p = ApplyTRS(verts[i], tr, root, poseInv);
                Vector3 nrmLocal = (norms != null && norms.Length == verts.Length) ? norms[i] : tr.up;
                Vector3 nrm = TransformNormal(nrmLocal, tr, root, poseInv);
                outPts.Add(p);
                outNrms.Add(nrm);
            }
            return;
        }

        for (int k = 0; k < count; k++)
        {
            float r = UnityEngine.Random.value * totalArea;
            int triIdx = 0; float acc = 0;
            for (; triIdx < triCount; triIdx++) { acc += triAreas[triIdx]; if (r <= acc) break; }
            triIdx = Mathf.Clamp(triIdx, 0, triCount - 1);

            int i0 = tris[3 * triIdx]; int i1 = tris[3 * triIdx + 1]; int i2 = tris[3 * triIdx + 2];
            var a = verts[i0]; var b = verts[i1]; var c = verts[i2];

            float u = UnityEngine.Random.value, v = UnityEngine.Random.value;
            if (u + v > 1f) { u = 1f - u; v = 1f - v; }
            var pLocal = a + u * (b - a) + v * (c - a);

            Vector3 nLocal;
            if (norms != null && norms.Length == verts.Length)
            {
                Vector3 n0 = norms[i0], n1 = norms[i1], n2 = norms[i2];
                nLocal = (n0 * (1 - u - v) + n1 * u + n2 * v).normalized;
            }
            else
            {
                nLocal = Vector3.Cross(b - a, c - a).normalized;
            }

            outPts.Add(ApplyTRS(pLocal, tr, root, poseInv));
            outNrms.Add(TransformNormal(nLocal, tr, root, poseInv));
        }
    }

    float ChamferWithNormalPenalty(
        List<Vector3> A, List<Vector3> nA,
        List<Vector3> B, List<Vector3> nB,
        float diag, float lambda)
    {
        int n = A.Count; if (n == 0 || B.Count == 0) return 0f;
        double sum = 0.0;

        for (int i = 0; i < n; i++)
        {
            Vector3 p = A[i];
            float bestSq = float.PositiveInfinity;
            int bestIdx = -1;
            for (int j = 0; j < B.Count; j++)
            {
                float dSq = (p - B[j]).sqrMagnitude;
                if (dSq < bestSq) { bestSq = dSq; bestIdx = j; }
            }
            float d = Mathf.Sqrt(bestSq);
            if (useCapped && diag > 1e-8f)
            {
                float cap = capAtDiagFrac * diag;
                if (d > cap) d = cap;
            }

            float dot = 1f;
            if (bestIdx >= 0 && nA != null && nB != null && nA.Count > i && nB.Count > bestIdx)
                dot = Mathf.Abs(Vector3.Dot(nA[i], nB[bestIdx]));

            float penalty = 1f + Mathf.Max(0f, lambda) * (1f - dot);
            sum += d * penalty;
        }
        return (float)(sum / n);
    }

    Vector3 TransformNormal(Vector3 nLocal, Transform child, Transform root, bool poseInv)
    {
        if (poseInv)
        {
            Quaternion q = Quaternion.Inverse(root.rotation) * child.rotation;
            return (q * nLocal).normalized;
        }
        else
        {
            return (child.rotation * nLocal).normalized;
        }
    }

    // ---------- Yardımcılar ----------
    Bounds BoundsFromPoints(List<Vector3> pts)
    {
        if (pts == null || pts.Count == 0) return new Bounds(Vector3.zero, Vector3.zero);
        var b = new Bounds(pts[0], Vector3.zero);
        for (int i = 1; i < pts.Count; i++) b.Encapsulate(pts[i]);
        return b;
    }

    void SampleFromObjectDeterministic(GameObject go, Transform root, int totalCount, bool poseInv, int baseSeed, List<Vector3> outPoints)
    {
        var filters = go.GetComponentsInChildren<MeshFilter>(includeInactive: true);
        if (filters == null || filters.Length == 0) return;

        Array.Sort(filters, (f1, f2) =>
        {
            string p1 = GetHierarchyPath(f1.transform);
            string p2 = GetHierarchyPath(f2.transform);
            return string.CompareOrdinal(p1, p2);
        });

        var pairs = new List<(Mesh mesh, Transform t)>(filters.Length);
        foreach (var f in filters) if (f.sharedMesh != null) pairs.Add((f.sharedMesh, f.transform));
        if (pairs.Count == 0) return;

        int childCount = pairs.Count;
        int baseN = Mathf.Max(1, totalCount / childCount);
        int remainder = Mathf.Max(0, totalCount - baseN * childCount);

        for (int i = 0; i < childCount; i++)
        {
            var p = pairs[i];
            int n = baseN + (i < remainder ? 1 : 0);

            int seed = CombineSeed(baseSeed, p.mesh.GetInstanceID(), p.t.GetInstanceID());
            var prevState = UnityEngine.Random.state;
            UnityEngine.Random.InitState(seed);

            SampleSurfacePoints(p.mesh, p.t, root, n, poseInv, outPoints);

            UnityEngine.Random.state = prevState;
        }
    }

    int CombineSeed(int a, int b, int c, int salt = 0)
    {
        unchecked
        {
            int x = a;
            x = x * 486187739 + b;
            x = x * 743398733 + c;
            x = x ^ (salt * 16777619);
            return x;
        }
    }

    string GetHierarchyPath(Transform t)
    {
        var sb = new StringBuilder(t.name);
        var cur = t.parent;
        while (cur != null)
        {
            sb.Insert(0, cur.name + "/");
            cur = cur.parent;
        }
        return sb.ToString();
    }



    void SampleSurfacePoints(Mesh mesh, Transform tr, Transform root, int count, bool poseInv, List<Vector3> outPoints)
    {
        var verts = mesh.vertices;
        var tris = mesh.triangles;
        int triCount = tris.Length / 3;

        if (triCount == 0 || verts == null || verts.Length == 0) return;

        var triAreas = new float[triCount];
        float totalArea = 0f;
        for (int i = 0; i < triCount; i++)
        {
            var v0 = verts[tris[3 * i]];
            var v1 = verts[tris[3 * i + 1]];
            var v2 = verts[tris[3 * i + 2]];
            float area = Vector3.Cross(v1 - v0, v2 - v0).magnitude * 0.5f;
            triAreas[i] = area;
            totalArea += area;
        }

        if (totalArea <= 1e-10f)
        {
            int n = Mathf.Min(count, verts.Length);
            for (int i = 0; i < n; i++)
                outPoints.Add(ApplyTRS(verts[i], tr, root, poseInv));
            return;
        }

        for (int k = 0; k < count; k++)
        {
            float r = UnityEngine.Random.value * totalArea;
            int triIdx = 0; float acc = 0;
            for (; triIdx < triCount; triIdx++)
            {
                acc += triAreas[triIdx];
                if (r <= acc) break;
            }
            triIdx = Mathf.Clamp(triIdx, 0, triCount - 1);

            var a = verts[tris[3 * triIdx]];
            var b = verts[tris[3 * triIdx + 1]];
            var c = verts[tris[3 * triIdx + 2]];

            float u = UnityEngine.Random.value, v = UnityEngine.Random.value;
            if (u + v > 1f) { u = 1f - u; v = 1f - v; }
            var pLocal = a + u * (b - a) + v * (c - a);

            outPoints.Add(ApplyTRS(pLocal, tr, root, poseInv));
        }
    }

    Vector3 ApplyTRS(Vector3 pLocal, Transform child, Transform root, bool poseInv)
    {
        if (poseInv)
        {
            Vector3 world = child.TransformPoint(pLocal);
            return root.InverseTransformPoint(world);
        }
        else
        {
            return child.TransformPoint(pLocal);
        }
    }

    // Outer/holes alan-çevre + hole centroid listelerini üretir.
    // ÇIKTI: holes, ALANA GÖRE BÜYÜKTEN KÜÇÜĞE SIRALANMIŞTIR (areas/perims/centroids paralel).
    void ComputeHoleStatsForObject(GameObject go, bool poseInv,
        out float outerArea, out float outerPerim,
        out List<float> holeAreas, out List<float> holePerims, out List<Vector3> holeCentroids)
    {
        outerArea = 0f; outerPerim = 0f;
        holeAreas = new List<float>();
        holePerims = new List<float>();
        holeCentroids = new List<Vector3>();

        // 1) Boundary segmentleri topla
        var segsRaw = new List<Seg>();
        foreach (var f in go.GetComponentsInChildren<MeshFilter>(true))
            if (f.sharedMesh) CollectBoundarySegments(f.sharedMesh, f.transform, go.transform, poseInv, segsRaw);
        if (segsRaw.Count == 0) return;

        // 2) Outer / hole sınıflandır
        List<Seg> segsOuter, segsHole;
        ClassifyLoops(segsRaw, out segsOuter, out segsHole);

        // 3) Loop’ları kur
        var loopsOuter = BuildLoopsFromSegments(segsOuter);
        var loopsHole = BuildLoopsFromSegments(segsHole);

        // 4) En büyük outer loop
        int bestIdx = -1; float bestAbs = 0f;
        for (int i = 0; i < loopsOuter.Count; i++)
        {
            float a = Mathf.Abs(SignedArea2D(loopsOuter[i]));
            if (a > bestAbs) { bestAbs = a; bestIdx = i; }
        }
        if (bestIdx >= 0)
        {
            outerArea = Mathf.Abs(SignedArea2D(loopsOuter[bestIdx]));
            outerPerim = PolylinePerimeter(loopsOuter[bestIdx]);
        }

        // 5) Hole istatistikleri
        var tmp = new List<(float area, float perim, Vector3 centroid)>();
        for (int i = 0; i < loopsHole.Count; i++)
        {
            var L = loopsHole[i];
            if (L.Count < 3) continue;
            float a = Mathf.Abs(SignedArea2D(L));
            float p = PolylinePerimeter(L);
            if (a <= 1e-8f || p <= 1e-6f) continue;

            // centroid (XY düzlem varsayımı: önceden ProjectToBestPlane çağrısı yapılmış olabilir)
            Vector3 c = PolygonCentroidXY(L);
            tmp.Add((a, p, c));
        }

        // 6) Alan büyükten küçüğe sırala ve paralel listeleri doldur
        tmp.Sort((u, v) => v.area.CompareTo(u.area));
        for (int i = 0; i < tmp.Count; i++)
        {
            holeAreas.Add(tmp[i].area);
            holePerims.Add(tmp[i].perim);
            holeCentroids.Add(tmp[i].centroid);
        }
    }

    float PolylinePerimeter(List<Vector3> poly)
    {
        float p = 0f;
        for (int i = 0; i < poly.Count; i++)
            p += Vector3.Distance(poly[i], poly[(i + 1) % poly.Count]);
        return p;
    }

    // Kapalı basit poligon centroid (XY düzleminde)
    Vector3 PolygonCentroidXY(List<Vector3> poly)
    {
        double A = 0.0, Cx = 0.0, Cy = 0.0;
        int n = poly.Count;
        for (int i = 0; i < n; i++)
        {
            var p = poly[i]; var q = poly[(i + 1) % n];
            double cross = (double)p.x * q.y - (double)q.x * p.y;
            A += cross;
            Cx += (p.x + q.x) * cross;
            Cy += (p.y + q.y) * cross;
        }
        A *= 0.5;
        if (Mathf.Abs((float)A) < 1e-12f) return new Vector3(0, 0, 0);
        Cx /= (6.0 * A);
        Cy /= (6.0 * A);
        return new Vector3((float)Cx, (float)Cy, 0f);
    }



    float AvgNearestDistRobust(
        List<Vector3> A, List<Vector3> B, float diag,
        float trimTopPercent, float capFrac,
        List<NNLine> debugLines, int maxLines,
        out float recall, float matchDeltaFrac)
    {
        recall = 0f;
        int n = A.Count;
        if (n == 0 || B.Count == 0) return 0f;

        debugLines?.Clear();

        var distsPow = new List<float>(n);
        int lineCounter = 0;
        float cap = (capFrac > 0f) ? (capFrac * diag) : float.PositiveInfinity;
        float matchDelta = Mathf.Max(1e-8f, matchDeltaFrac * diag);

        for (int i = 0; i < n; i++)
        {
            Vector3 p = A[i];
            float bestSq = float.PositiveInfinity;
            int bestIdx = -1;

            for (int j = 0; j < B.Count; j++)
            {
                float dSq = (p - B[j]).sqrMagnitude;
                if (dSq < bestSq) { bestSq = dSq; bestIdx = j; }
            }

            float d = Mathf.Sqrt(bestSq);
            if (d > cap) d = cap;               // capped
            distsPow.Add(Mathf.Pow(d, distanceExponent));

            if (debugLines != null && lineCounter < maxLines && bestIdx >= 0)
            {
                debugLines.Add(new NNLine(p, B[bestIdx]));
                lineCounter++;
            }
        }

        // recall
        int hits = 0;
        for (int i = 0; i < distsPow.Count; i++)
        {
            float d = Mathf.Pow(distsPow[i], 1f / distanceExponent);
            if (d <= matchDelta) hits++;
        }
        recall = (float)hits / distsPow.Count;

        // trimmed mean (pow domaininde ortala, sonra kök)
        distsPow.Sort();
        if (trimTopPercent > 0f)
        {
            int keep = Mathf.Max(1, Mathf.RoundToInt(distsPow.Count * (1f - Mathf.Clamp01(trimTopPercent))));
            double sum = 0.0;
            for (int i = 0; i < keep; i++) sum += distsPow[i];
            float meanPow = (float)(sum / keep);
            return Mathf.Pow(meanPow, 1f / distanceExponent);
        }
        else
        {
            double sum = 0.0;
            for (int i = 0; i < distsPow.Count; i++) sum += distsPow[i];
            float meanPow = (float)(sum / distsPow.Count);
            return Mathf.Pow(meanPow, 1f / distanceExponent);
        }
    }

    float PercentileNearestDist(
        List<Vector3> A, List<Vector3> B, float diag,
        float capFrac, out float recall, float matchDeltaFrac, float percentile)
    {
        recall = 0f; if (A.Count == 0 || B.Count == 0) return 0f;

        var distsPow = new List<float>(A.Count);
        float cap = (capFrac > 0f) ? (capFrac * diag) : float.PositiveInfinity;
        float matchDelta = Mathf.Max(1e-8f, matchDeltaFrac * diag);

        for (int i = 0; i < A.Count; i++)
        {
            var p = A[i]; float bestSq = float.PositiveInfinity; int bestIdx = -1;
            for (int j = 0; j < B.Count; j++)
            {
                float dsq = (p - B[j]).sqrMagnitude;
                if (dsq < bestSq) { bestSq = dsq; bestIdx = j; }
            }
            float d = Mathf.Sqrt(bestSq);
            if (d > cap) d = cap;
            distsPow.Add(Mathf.Pow(d, distanceExponent));
        }

        // recall
        int hits = 0;
        for (int i = 0; i < distsPow.Count; i++)
        {
            float d = Mathf.Pow(distsPow[i], 1f / distanceExponent);
            if (d <= matchDelta) hits++;
        }
        recall = (float)hits / distsPow.Count;

        // percentile (pow domaininde seç, sonra kök)
        distsPow.Sort();
        int k = Mathf.Clamp(Mathf.RoundToInt((distsPow.Count - 1) * Mathf.Clamp01(percentile)), 0, distsPow.Count - 1);
        return Mathf.Pow(distsPow[k], 1f / distanceExponent);
    }

    float CombinedWorldDiag(GameObject a, GameObject b)
    {
        Bounds ba = WorldBoundsOf(a);
        Bounds bb = WorldBoundsOf(b);
        Vector3 min = Vector3.Min(ba.min, bb.min);
        Vector3 max = Vector3.Max(ba.max, bb.max);
        return (max - min).magnitude;
    }

    Bounds WorldBoundsOf(GameObject go)
    {
        var rends = go.GetComponentsInChildren<Renderer>(includeInactive: true);
        if (rends.Length == 0) return new Bounds(go.transform.position, Vector3.zero);
        var b = rends[0].bounds;
        for (int i = 1; i < rends.Length; i++) b.Encapsulate(rends[i].bounds);
        return b;
    }



    Bounds LocalMeshBoundsAll(GameObject go)
    {
        var filters = go.GetComponentsInChildren<MeshFilter>(includeInactive: true);
        bool hasAny = false;
        Bounds b = new Bounds(Vector3.zero, Vector3.zero);

        Array.Sort(filters, (x, y) =>
            string.CompareOrdinal(GetHierarchyPath(x.transform), GetHierarchyPath(y.transform)));

        foreach (var f in filters)
        {
            if (f.sharedMesh == null) continue;
            var mb = LocalMeshBounds(f.sharedMesh);
            if (!hasAny) { b = mb; hasAny = true; }
            else b.Encapsulate(mb);
        }
        if (!hasAny) b = new Bounds(Vector3.zero, Vector3.zero);
        return b;
    }

    Bounds LocalMeshBounds(Mesh m)
    {
        var v = m.vertices;
        if (v == null || v.Length == 0) return new Bounds(Vector3.zero, Vector3.zero);
        var b = new Bounds(v[0], Vector3.zero);
        for (int i = 1; i < v.Length; i++) b.Encapsulate(v[i]);
        return b;
    }

    void SpawnMarkers()
    {
        if (markersRoot == null)
        {
            var root = new GameObject("PointMarkers");
            markersRoot = root.transform;
        }

        for (int i = markersRoot.childCount - 1; i >= 0; i--)
        {
            if (Application.isPlaying) Destroy(markersRoot.GetChild(i).gameObject);
            else DestroyImmediate(markersRoot.GetChild(i).gameObject);
        }

        if (pointPrefab == null) return;

        var aRoot = new GameObject("A_Points").transform; aRoot.parent = markersRoot;
        foreach (var pLocal in ptsA)
        {
            Vector3 pWorld = poseInvariant ? objectA.transform.TransformPoint(pLocal) : pLocal; // <<< DÜZ.
            Instantiate(pointPrefab, pWorld, Quaternion.identity, aRoot).name = "Apt";
        }

        var bRoot = new GameObject("B_Points").transform; bRoot.parent = markersRoot;
        foreach (var pLocal in ptsB)
        {
            Vector3 pWorld = poseInvariant ? objectB.transform.TransformPoint(pLocal) : pLocal; // <<< DÜZ.
            Instantiate(pointPrefab, pWorld, Quaternion.identity, bRoot).name = "Bpt";
        }
    }


    // -------- Gizmos --------
    void OnDrawGizmos()
    {
        if (!drawGizmos) return;
        if (!drawWhenUnselected) return;
        DrawGizmosCommon();
    }
    void OnDrawGizmosSelected()
    {
        if (!drawGizmos) return;
        if (drawWhenUnselected) return;
        DrawGizmosCommon();
    }
    void DrawGizmosCommon()
    {
        if (objectA == null || objectB == null) return;
        float r = pointSize;

        // A noktaları
        Gizmos.color = new Color(0.1f, 0.7f, 1f, 0.9f);
        for (int i = 0; i < ptsA.Count; i++)
        {
            Vector3 p = ptsA[i];
            if (poseInvariant) p = objectA.transform.TransformPoint(p); // <<< DÜZELTİLDİ
            Gizmos.DrawSphere(p, r);
        }

        // B noktaları
        Gizmos.color = new Color(1f, 0.4f, 0.1f, 0.9f);
        for (int i = 0; i < ptsB.Count; i++)
        {
            Vector3 p = ptsB[i];
            if (poseInvariant) p = objectB.transform.TransformPoint(p); // <<< DÜZELTİLDİ
            Gizmos.DrawSphere(p, r);
        }

        // NN çizgileri
        if (drawNearestNeighborLines)
        {
            Gizmos.color = new Color(0.1f, 1f, 0.3f, 0.9f);
            for (int i = 0; i < nnLinesA2B.Count; i++)
            {
                var l = nnLinesA2B[i];
                Vector3 from = poseInvariant ? objectA.transform.TransformPoint(l.from) : l.from; // <<< DÜZ.
                Vector3 to = poseInvariant ? objectB.transform.TransformPoint(l.to) : l.to;   // <<< DÜZ.
                Gizmos.DrawLine(from, to);
            }

            Gizmos.color = new Color(1f, 0.2f, 0.7f, 0.9f);
            for (int i = 0; i < nnLinesB2A.Count; i++)
            {
                var l = nnLinesB2A[i];
                Vector3 from = poseInvariant ? objectB.transform.TransformPoint(l.from) : l.from; // <<< DÜZ.
                Vector3 to = poseInvariant ? objectA.transform.TransformPoint(l.to) : l.to;   // <<< DÜZ.
                Gizmos.DrawLine(from, to);
            }
        }

        if (objectA && objectB && !poseInvariant)
        {
            var ba = WorldBoundsOf(objectA);
            var bb = WorldBoundsOf(objectB);
            Vector3 min = Vector3.Min(ba.min, bb.min);
            Vector3 max = Vector3.Max(ba.max, bb.max);
            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(min, max);
        }
    }



}

#if UNITY_EDITOR
[CustomPropertyDrawer(typeof(ReadOnlyAttribute))]
public class ReadOnlyDrawer : PropertyDrawer
{
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        GUI.enabled = false;
        EditorGUI.PropertyField(position, property, label, true);
        GUI.enabled = true;
    }
}
#endif
