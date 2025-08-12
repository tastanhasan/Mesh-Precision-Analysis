using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[ExecuteAlways]
public class MeshSimilarityLite : MonoBehaviour
{
    [Header("Targets")]
    public GameObject objectA;
    public GameObject objectB;

    [Header("Mode")]
    public bool perChildMode = false;        // false: Global, true: Per-Child
    public bool reportWorstChild = true;     // Per-Child modda en kötü çocuk skorunu da yaz

    [Header("Sampling")]
    [Range(500, 50000)] public int samplesPerMesh = 8000;
    public bool poseInvariant = true;        // Global modda root-local kıyas
    public int randomSeed = 12345;

    [Header("Robustness")]
    [Range(0f, 0.5f)] public float trimTopPercent = 0.10f;   // en uzak %X at
    [Range(0f, 0.5f)] public float capAtDiagFrac = 0.10f;    // d <= cap*diag
    [Range(0f, 0.2f)] public float matchDeltaFrac = 0.02f;   // recall eşiği

    [Header("Pose Alignment (ICP)")]
    public bool useICPAlign = true;          // Global: tüm bulut; Per-Child: her çocuk çifti
    [Range(1, 30)] public int icpIterations = 8;
    [Range(0f, 0.5f)] public float icpTrimPercent = 0.20f;
    [Range(200, 20000)] public int icpWorkingSize = 2000;

    [Header("Rotation Penalty (Per-Child)")]
    public bool useRotationPenalty = false;          // Per-Child’ta rotasyon farkını da hesaba kat
    [Range(0f, 1f)] public float rotWeight = 0.5f;   // 0: sadece şekil, 1: sadece rotasyon
    [Range(1f, 180f)] public float rotCapDeg = 90f;  // 90° ve üstü tam ceza

    [Header("Gizmos")]
    public bool drawGizmos = true;
    public bool drawNNLines = false;         // sadece Global modda çiziyoruz
    [Range(10, 2000)] public int maxNNLines = 200;
    [Range(0.002f, 0.05f)] public float pointSize = 0.01f;
    public enum RotPenaltySource { None, ParentLocal, ICPApplied }
    [Header("Rotation Penalty (Per‑Child)")]
    public RotPenaltySource rotationPenaltySource = RotPenaltySource.ParentLocal;

    public float rotDeadzoneDeg = 2f; // küçük açıyı yok say

    [Header("Outputs (read-only)")]
    [SerializeField, ReadOnly] float similarity01;
    [SerializeField, ReadOnly] float chamfer;
    [SerializeField, ReadOnly] float diagonal;
    [SerializeField, ReadOnly] float precision;
    [SerializeField, ReadOnly] float recall;
    [SerializeField, ReadOnly] float f1;
    [SerializeField, ReadOnly] float worstChildSimilarity01;

    // internal state
    readonly List<Vector3> ptsA = new List<Vector3>();
    readonly List<Vector3> ptsB = new List<Vector3>();
    readonly List<(Vector3 from, Vector3 to)> nnAB = new List<(Vector3, Vector3)>();
    readonly List<(Vector3 from, Vector3 to)> nnBA = new List<(Vector3, Vector3)>();

    void Start() { if (Application.isPlaying) Compute(); }
    void Update() { if (Input.GetKeyDown(KeyCode.R)) Compute(); }

    [ContextMenu("Compute")]
    [ContextMenu("Compute")]
    public void Compute()
    {
        if (objectA == null || objectB == null)
        {
            UnityEngine.Debug.LogWarning("[MeshSimLite] Object A/B atayın.");
            return;
        }

        var swTotal = Stopwatch.StartNew();
        var rngState = Random.state;

        if (!perChildMode)
        {
            // ---------- GLOBAL (ROBUST) ----------
            ptsA.Clear(); ptsB.Clear();
            SampleFromObject(objectA, objectA.transform, samplesPerMesh, poseInvariant, randomSeed, ptsA);
            SampleFromObject(objectB, objectB.transform, samplesPerMesh, poseInvariant, randomSeed, ptsB);
            if (ptsA.Count == 0 || ptsB.Count == 0) { UnityEngine.Debug.LogError("[MeshSimLite] Nokta bulutu boş."); return; }

            if (poseInvariant && useICPAlign)
            {
                float pre = MeanNN(ptsA, ptsB);
                ICPAlignInPlace(ptsA, ptsB, icpIterations, icpWorkingSize, icpTrimPercent);
                float post = MeanNN(ptsA, ptsB);
                UnityEngine.Debug.Log($"[MeshSimLite] ICP(Global) meanNN: {pre:F4} -> {post:F4}");
            }

            diagonal = BoundsDiag(ptsA, ptsB);
            if (diagonal <= 1e-8f) diagonal = 1e-8f;

            nnAB.Clear(); nnBA.Clear();
            float rAB, rBA;

            float chAB = AvgNearestDistRobust(
                ptsA, ptsB, diagonal, trimTopPercent, capAtDiagFrac,
                drawNNLines ? nnAB : null, maxNNLines, out rAB, matchDeltaFrac);

            float chBA = AvgNearestDistRobust(
                ptsB, ptsA, diagonal, trimTopPercent, capAtDiagFrac,
                drawNNLines ? nnBA : null, maxNNLines, out rBA, matchDeltaFrac);

            chamfer = 0.5f * (chAB + chBA);
            similarity01 = 1f - Mathf.Clamp01(chamfer / diagonal);

            recall = 0.5f * (rAB + rBA);
            precision = recall;
            f1 = (precision + recall > 1e-6f) ? 2f * precision * recall / (precision + recall) : 0f;

            swTotal.Stop();
            UnityEngine.Debug.Log(
                $"[MeshSimilarity] Mode: Global (Robust)\n" +
                $"- Samples per mesh: {samplesPerMesh}\n" +
                $"- CH (equiv): {chamfer:F6}\n" +
                $"- Diagonal: {diagonal:F6}\n" +
                $"- Similarity (0..1): {similarity01:F6}\n" +
                $"- Recall(A→B,B→A,avg): {rAB:P1}, {rBA:P1}, {recall:P1}\n" +
                $"- F1: {f1:P1}\n" +
                $"- Total: {swTotal.ElapsedMilliseconds} ms");
        }
        else
        {
            // ---------- PER-CHILD ----------
            var fa = objectA.GetComponentsInChildren<MeshFilter>(true);
            var fb = objectB.GetComponentsInChildren<MeshFilter>(true);
            System.Array.Sort(fa, (x, y) => string.CompareOrdinal(Path(x.transform), Path(y.transform)));
            System.Array.Sort(fb, (x, y) => string.CompareOrdinal(Path(x.transform), Path(y.transform)));

            int m = Mathf.Min(fa.Length, fb.Length);
            if (m == 0) { UnityEngine.Debug.LogWarning("[MeshSimLite] Per-Child: çocuk mesh bulunamadı."); return; }

            int perChild = Mathf.Max(200, samplesPerMesh / m);

            double sumSim = 0.0;
            double sumRecall = 0.0;      // <-- recall artık toplanıyor
            float worstSim = 1f;

            for (int i = 0; i < m; i++)
            {
                var ma = fa[i].sharedMesh;
                var mb = fb[i].sharedMesh;
                if (ma == null || mb == null) { sumSim += 0; worstSim = 0; continue; }

                // Dünya koordinatında örnekle
                var pa = new List<Vector3>(perChild);
                var pb = new List<Vector3>(perChild);

                int seedA = CombineSeed(randomSeed, ma.GetInstanceID(), fa[i].transform.GetInstanceID());
                int seedB = CombineSeed(randomSeed, mb.GetInstanceID(), fb[i].transform.GetInstanceID());
                var prev = Random.state;

                Random.InitState(seedA);
                SampleSurfacePoints(ma, fa[i].transform, objectA.transform, perChild, false, pa); // world

                Random.InitState(seedB);
                SampleSurfacePoints(mb, fb[i].transform, objectB.transform, perChild, false, pb); // world

                Random.state = prev;

                // Çift bazında ICP hizalama + uygulanan rotasyonu al
                Quaternion icpRot = Quaternion.identity;
                if (useICPAlign)
                    ICPAlignInPlace(pa, pb, icpIterations,
                                    Mathf.Min(icpWorkingSize, perChild),
                                    icpTrimPercent,
                                    out icpRot);

                // Diagonal & chamfer
                float diag = BoundsDiag(pa, pb);
                if (diag <= 1e-8f) diag = 1e-8f;

                float rAB, rBA;
                float cAB = AvgNearestDistRobust(pa, pb, diag, trimTopPercent, capAtDiagFrac,
                                                 null, 0, out rAB, matchDeltaFrac);
                float cBA = AvgNearestDistRobust(pb, pa, diag, trimTopPercent, capAtDiagFrac,
                                                 null, 0, out rBA, matchDeltaFrac);

                float ch = 0.5f * (cAB + cBA);
                float shapeSim = 1f - Mathf.Clamp01(ch / diag);

                // --- Rotasyon cezası: ICP’nin uyguladığı gerçek açı ---
                float finalSim = shapeSim;
                if (useRotationPenalty && rotationPenaltySource != RotPenaltySource.None)
                {
                    float thetaDeg = 0f;

                    if (rotationPenaltySource == RotPenaltySource.ParentLocal)
                    {
                        // ebeveyne göre relatif rotasyon farkı (ICP’den bağımsız)
                        Quaternion qA = Quaternion.Inverse(objectA.transform.rotation) * fa[i].transform.rotation;
                        Quaternion qB = Quaternion.Inverse(objectB.transform.rotation) * fb[i].transform.rotation;
                        thetaDeg = Quaternion.Angle(qA, qB);
                    }
                    else // ICPApplied
                    {
                        thetaDeg = Quaternion.Angle(icpRot, Quaternion.identity);
                    }

                    if (thetaDeg < rotDeadzoneDeg) thetaDeg = 0f;

                    float rotSim = Mathf.Clamp01(1f - thetaDeg / Mathf.Max(1e-3f, rotCapDeg));
                    // çarpanlı ve monoton azalan kombinasyon (benzerliği asla arttırmaz)
                    float rotFactor = Mathf.Lerp(1f, rotSim, rotWeight); // 1..rotSim
                    finalSim = shapeSim * rotFactor;
                }


                // Toplamlara ekle
                sumSim += finalSim;
                sumRecall += 0.5f * (rAB + rBA);   // recall toplanıyor
                if (finalSim < worstSim) worstSim = finalSim;
            }

            similarity01 = (float)(sumSim / m);
            worstChildSimilarity01 = worstSim;
            recall = (float)(sumRecall / m);
            precision = recall;
            f1 = (precision + recall > 1e-6f) ? 2f * precision * recall / (precision + recall) : 0f;

            // rapor kolaylığı
            diagonal = 1f;
            chamfer = Mathf.Clamp01(1f - similarity01);

            swTotal.Stop();
            UnityEngine.Debug.Log(
                $"[MeshSimilarity] Mode: Per-Child\n" +
                $"- Children paired: {m}, per-child samples ≈ {perChild}\n" +
                $"- Similarity (mean): {similarity01:F6}  {(reportWorstChild ? $"| worst: {worstChildSimilarity01:F6}" : "")}\n" +
                $"- Avg Recall: {recall:P1}  | F1: {f1:P1}\n" +
                $"- Total: {swTotal.ElapsedMilliseconds} ms");
        }

        Random.state = rngState;
#if UNITY_EDITOR
        SceneView.RepaintAll();
#endif
    }

    static Quaternion QuaternionFromMatrix3x3(Matrix4x4 R)
    {
        Vector3 c0 = new Vector3(R.m00, R.m10, R.m20);
        Vector3 c1 = new Vector3(R.m01, R.m11, R.m21);
        Vector3 c2 = new Vector3(R.m02, R.m12, R.m22);
        c0 = c0.normalized;
        c1 = (c1 - Vector3.Dot(c1, c0) * c0).normalized;
        c2 = Vector3.Cross(c0, c1).normalized;

        Matrix4x4 M = Matrix4x4.identity;
        M.m00 = c0.x; M.m10 = c0.y; M.m20 = c0.z;
        M.m01 = c1.x; M.m11 = c1.y; M.m21 = c1.z;
        M.m02 = c2.x; M.m12 = c2.y; M.m22 = c2.z;
        return M.rotation;
    }

    // -------- Sampling (shared) --------
    void SampleFromObject(GameObject go, Transform root, int total, bool poseInv, int baseSeed, List<Vector3> outPts)
    {
        outPts.Clear();
        var filters = go.GetComponentsInChildren<MeshFilter>(true);
        if (filters == null || filters.Length == 0) return;

        System.Array.Sort(filters, (a, b) => string.CompareOrdinal(Path(a.transform), Path(b.transform)));
        var pairs = new List<(Mesh m, Transform t)>();
        foreach (var f in filters) if (f.sharedMesh) pairs.Add((f.sharedMesh, f.transform));
        if (pairs.Count == 0) return;

        int childCount = pairs.Count;
        int baseN = Mathf.Max(1, total / childCount);
        int rem = Mathf.Max(0, total - baseN * childCount);

        for (int i = 0; i < childCount; i++)
        {
            int n = baseN + (i < rem ? 1 : 0);
            int seed = CombineSeed(baseSeed, pairs[i].m.GetInstanceID(), pairs[i].t.GetInstanceID());
            var prev = Random.state; Random.InitState(seed);
            SampleSurfacePoints(pairs[i].m, pairs[i].t, root, n, poseInv, outPts);
            Random.state = prev;
        }
    }

    void SampleSurfacePoints(Mesh mesh, Transform tr, Transform root, int count, bool poseInv, List<Vector3> outPts)
    {
        var v = mesh.vertices; var t = mesh.triangles;
        int triCount = t.Length / 3;
        if (triCount == 0 || v == null || v.Length == 0) return;

        var areas = new float[triCount];
        float totalArea = 0f;
        for (int i = 0; i < triCount; i++)
        {
            var a = v[t[3 * i]]; var b = v[t[3 * i + 1]]; var c = v[t[3 * i + 2]];
            float area = Vector3.Cross(b - a, c - a).magnitude * 0.5f;
            areas[i] = area; totalArea += area;
        }

        if (totalArea <= 1e-10f)
        {
            int n = Mathf.Min(count, v.Length);
            for (int i = 0; i < n; i++) outPts.Add(ApplyTRS(v[i], tr, root, poseInv));
            return;
        }

        for (int k = 0; k < count; k++)
        {
            float r = Random.value * totalArea;
            int tri = 0; float acc = 0f;
            for (; tri < triCount; tri++) { acc += areas[tri]; if (r <= acc) break; }
            tri = Mathf.Clamp(tri, 0, triCount - 1);

            var a = v[t[3 * tri]]; var b = v[t[3 * tri + 1]]; var c = v[t[3 * tri + 2]];
            float u = Random.value, w = Random.value; if (u + w > 1f) { u = 1f - u; w = 1f - w; }
            var pLocal = a + u * (b - a) + w * (c - a);

            outPts.Add(ApplyTRS(pLocal, tr, root, poseInv));
        }
    }

    Vector3 ApplyTRS(Vector3 pLocal, Transform child, Transform root, bool poseInv)
    {
        if (poseInv) { var world = child.TransformPoint(pLocal); return root.InverseTransformPoint(world); }
        else return child.TransformPoint(pLocal);
    }

    // -------- Robust NN ----------
    float AvgNearestDistRobust(
        List<Vector3> A, List<Vector3> B, float diag,
        float trimTopPercent, float capFrac,
        List<(Vector3, Vector3)> debugLines, int maxLines,
        out float recall, float matchDeltaFrac)
    {
        recall = 0f;
        int n = A.Count; if (n == 0 || B.Count == 0) return 0f;
        if (debugLines != null) debugLines.Clear();

        var dists = new List<float>(n);
        float cap = (capFrac > 0f) ? (capFrac * diag) : float.PositiveInfinity;
        float matchDelta = Mathf.Max(1e-8f, matchDeltaFrac * diag);
        int lines = 0;

        for (int i = 0; i < n; i++)
        {
            Vector3 p = A[i];
            float bestSq = float.PositiveInfinity; int bestIdx = -1;
            for (int j = 0; j < B.Count; j++)
            {
                float dsq = (p - B[j]).sqrMagnitude;
                if (dsq < bestSq) { bestSq = dsq; bestIdx = j; }
            }
            float d = Mathf.Sqrt(bestSq);
            if (d > cap) d = cap;
            dists.Add(d);

            if (debugLines != null && lines < maxLines && bestIdx >= 0)
            {
                debugLines.Add((p, B[bestIdx]));
                lines++;
            }
        }

        int hits = 0; for (int i = 0; i < dists.Count; i++) if (dists[i] <= matchDelta) hits++;
        recall = (float)hits / dists.Count;

        if (trimTopPercent > 0f)
        {
            dists.Sort();
            int keep = Mathf.Max(1, Mathf.RoundToInt(dists.Count * (1f - Mathf.Clamp01(trimTopPercent))));
            double sum = 0.0; for (int i = 0; i < keep; i++) sum += dists[i];
            return (float)(sum / keep);
        }
        else
        {
            double sum = 0.0; for (int i = 0; i < dists.Count; i++) sum += dists[i];
            return (float)(sum / dists.Count);
        }
    }

    // -------- ICP hizalama (A’yı B’ye) --------
    void ICPAlignInPlace(List<Vector3> A, List<Vector3> B, int iters, int workN, float trimTop, out Quaternion totalRot)
    {
        totalRot = Quaternion.identity;
        if (A.Count == 0 || B.Count == 0) return;

        int nA = Mathf.Min(workN, A.Count), nB = Mathf.Min(workN, B.Count);
        var a = new List<Vector3>(nA); var b = new List<Vector3>(nB);
        for (int i = 0; i < nA; i++) a.Add(A[i * A.Count / nA]);
        for (int i = 0; i < nB; i++) b.Add(B[i * B.Count / nB]);

        // merkez eşitle
        Vector3 ca = Centroid(a), cb = Centroid(b);
        Vector3 t0 = cb - ca;
        for (int i = 0; i < A.Count; i++) A[i] += t0;
        for (int i = 0; i < a.Count; i++) a[i] += t0;

        for (int it = 0; it < iters; it++)
        {
            var PA = new List<Vector3>(a.Count);
            var PB = new List<Vector3>(a.Count);
            for (int i = 0; i < a.Count; i++)
            {
                float best = float.PositiveInfinity; int idx = -1;
                for (int j = 0; j < b.Count; j++)
                {
                    float dsq = (a[i] - b[j]).sqrMagnitude;
                    if (dsq < best) { best = dsq; idx = j; }
                }
                PA.Add(a[i]); PB.Add(b[idx]);
            }

            if (trimTop > 0f && PA.Count > 10)
            {
                var order = new List<int>(PA.Count);
                for (int i = 0; i < PA.Count; i++) order.Add(i);
                order.Sort((i, j) => ((PA[i] - PB[i]).sqrMagnitude).CompareTo((PA[j] - PB[j]).sqrMagnitude));
                int keep = Mathf.Max(10, Mathf.RoundToInt(order.Count * (1f - Mathf.Clamp01(trimTop))));
                var a2 = new List<Vector3>(keep); var b2 = new List<Vector3>(keep);
                for (int k = 0; k < keep; k++) { a2.Add(PA[order[k]]); b2.Add(PB[order[k]]); }
                PA = a2; PB = b2;
            }

            Matrix4x4 R = Kabsch(PA, PB);
            Quaternion q = QuaternionFromMatrix3x3(R);
            // sıralı uygulamalar için toplam = R_it * toplam
            totalRot = q * totalRot;

            for (int i = 0; i < a.Count; i++) a[i] = R.MultiplyPoint3x4(a[i]);
            for (int i = 0; i < A.Count; i++) A[i] = R.MultiplyPoint3x4(A[i]);
        }
    }

    // Eski imza çağrılarını korumak için kısa yol:
    void ICPAlignInPlace(List<Vector3> A, List<Vector3> B, int iters, int workN, float trimTop)
    {
        ICPAlignInPlace(A, B, iters, workN, trimTop, out _);
    }


    Matrix4x4 Kabsch(List<Vector3> A, List<Vector3> B)
    {
        float xx = 0, xy = 0, xz = 0, yx = 0, yy = 0, yz = 0, zx = 0, zy = 0, zz = 0;
        for (int i = 0; i < A.Count; i++)
        {
            var a = A[i]; var b = B[i];
            xx += a.x * b.x; xy += a.x * b.y; xz += a.x * b.z;
            yx += a.y * b.x; yy += a.y * b.y; yz += a.y * b.z;
            zx += a.z * b.x; zy += a.z * b.y; zz += a.z * b.z;
        }
        // kolonları ortonormalize et (polar decomposition approx)
        Vector3 c0 = new Vector3(xx, yx, zx).normalized;
        Vector3 v1 = new Vector3(xy, yy, zy);
        Vector3 c1 = (v1 - Vector3.Dot(v1, c0) * c0).normalized;
        Vector3 c2 = Vector3.Cross(c0, c1).normalized;
        Matrix4x4 R = Matrix4x4.identity;
        R.m00 = c0.x; R.m10 = c0.y; R.m20 = c0.z;
        R.m01 = c1.x; R.m11 = c1.y; R.m21 = c1.z;
        R.m02 = c2.x; R.m12 = c2.y; R.m22 = c2.z;
        return R;
    }

    // -------- utils --------
    static string Path(Transform t) { var s = t.name; var p = t.parent; while (p) { s = p.name + "/" + s; p = p.parent; } return s; }
    static int CombineSeed(int a, int b, int c) { unchecked { int x = a; x = x * 486187739 + b; x = x * 743398733 + c; return x; } }
    static Vector3 Centroid(List<Vector3> P) { Vector3 s = Vector3.zero; for (int i = 0; i < P.Count; i++) s += P[i]; return s / P.Count; }
    static float BoundsDiag(List<Vector3> A, List<Vector3> B) { var ba = BoundsOf(A); var bb = BoundsOf(B); var min = Vector3.Min(ba.min, bb.min); var max = Vector3.Max(ba.max, bb.max); return (max - min).magnitude; }
    static Bounds BoundsOf(List<Vector3> P) { if (P.Count == 0) return new Bounds(Vector3.zero, Vector3.zero); var b = new Bounds(P[0], Vector3.zero); for (int i = 1; i < P.Count; i++) b.Encapsulate(P[i]); return b; }
    static float MeanNN(List<Vector3> A, List<Vector3> B) { if (A.Count == 0 || B.Count == 0) return 0f; double s = 0; for (int i = 0; i < A.Count; i++) { float best = float.PositiveInfinity; for (int j = 0; j < B.Count; j++) { float d = (A[i] - B[j]).sqrMagnitude; if (d < best) best = d; } s += Mathf.Sqrt(best); } return (float)(s / A.Count); }

    // -------- Gizmos --------
    void OnDrawGizmosSelected() { if (!drawGizmos) return; Draw(); }
    void OnDrawGizmos() { if (!drawGizmos) return; Draw(); }
    void Draw()
    {
        float r = pointSize;
        Gizmos.color = new Color(0.1f, 0.7f, 1f, 0.9f);
        for (int i = 0; i < ptsA.Count; i++) Gizmos.DrawSphere(ptsA[i], r);
        Gizmos.color = new Color(1f, 0.4f, 0.1f, 0.9f);
        for (int i = 0; i < ptsB.Count; i++) Gizmos.DrawSphere(ptsB[i], r);

        if (!perChildMode && drawNNLines)
        {
            Gizmos.color = new Color(0.1f, 1f, 0.3f, 0.9f);
            for (int i = 0; i < nnAB.Count; i++) Gizmos.DrawLine(nnAB[i].from, nnAB[i].to);
            Gizmos.color = new Color(1f, 0.2f, 0.7f, 0.9f);
            for (int i = 0; i < nnBA.Count; i++) Gizmos.DrawLine(nnBA[i].from, nnBA[i].to);
        }
    }

    // Inspector'da read-only
    class ReadOnlyAttribute : PropertyAttribute { }
#if UNITY_EDITOR
    [CustomPropertyDrawer(typeof(ReadOnlyAttribute))]
    class ReadOnlyDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect pos, SerializedProperty prop, GUIContent label)
        { GUI.enabled = false; EditorGUI.PropertyField(pos, prop, label, true); GUI.enabled = true; }
    }
#endif
}
