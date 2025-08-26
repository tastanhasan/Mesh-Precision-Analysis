// MSDPresetApplier.cs
using System.Collections;
using System.Linq;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// Play başlamadan önce ve/veya Edit Mode'da MeshSimilarityDebugger ayarlarını
/// iki hazır presete göre uygular. İstersen Edit Mode'da da ComputeAll çağırır.
/// </summary>
[ExecuteAlways]
[DefaultExecutionOrder(-1000)]
public class MSDPresetApplier : MonoBehaviour
{
    [Tooltip("Hedef MeshSimilarityDebugger. Boşsa sahnede otomatik aranır.")]
    public MeshSimilarityDebugger msd;

    public enum Preset { None, Task3, Task1, Task2 }   // <— None eklendi

    [Header("Preset")]
    public Preset presetToApply = Preset.Task2;

    [Header("Auto Apply (Play)")]
    [Tooltip("Play moduna girerken preset uygula (sahne yeniden yüklense bile).")]
    public bool applyOnEnterPlay = true;

    [Tooltip("Play başladıktan bir sonraki frame'de ComputeAll çağır.")]
    public bool computeAfterApply = true;

    [Header("Auto Apply (Edit)")]
    [Tooltip("Edit Mode'da, bileşen enable olduğunda veya değerleri değiştiğinde preset uygula.")]
    public bool applyInEditPreview = false;

    [Tooltip("Edit Mode'da preset uygulandıktan sonra ComputeAll çağır.")]
    public bool computeInEditPreview = false;

#if UNITY_EDITOR
    bool _pendingApplyEdit;   // OnValidate debounce
#endif

    // -------- Lifecycle --------
    void Reset() => TryResolveMSD();

    void OnEnable()
    {
        TryResolveMSD();

#if UNITY_EDITOR
        if (!Application.isPlaying && applyInEditPreview)
            ScheduleEditApply(computeInEditPreview);
#endif
    }

    void Awake()
    {
        TryResolveMSD();
        if (Application.isPlaying && applyOnEnterPlay)
            ApplySelected(editTime: false, alsoCompute: false);
    }

    IEnumerator Start()
    {
        if (Application.isPlaying && computeAfterApply && msd != null)
        {
            // Tüm Awake/Start zinciri tamamlansın
            yield return null;
            msd.ComputeAll();
        }
    }

#if UNITY_EDITOR
    void OnValidate()
    {
        if (Application.isPlaying) return;

        // Manuel modda kesinlikle tetikleme
        if (!applyInEditPreview || presetToApply == Preset.None) return;


        // Inspector değişikliklerini bir sonraki editor döngüsünde uygula
        ScheduleEditApply(computeInEditPreview);
    }

    void ScheduleEditApply(bool alsoCompute)
    {
        if (_pendingApplyEdit) return;
        _pendingApplyEdit = true;

        EditorApplication.delayCall += () =>
        {
            if (this == null) { _pendingApplyEdit = false; return; }
            ApplySelected(editTime: true, alsoCompute: alsoCompute);
            _pendingApplyEdit = false;
        };
    }
#endif

    // -------- Uygulama --------
    void TryResolveMSD()
    {
        if (msd) return;

        // Aynı objede var mı?
        msd = GetComponent<MeshSimilarityDebugger>();
        if (msd) return;

        // Sahnede bul (inactive dahil)
#if UNITY_2020_1_OR_NEWER
        msd = FindObjectOfType<MeshSimilarityDebugger>(true);
#else
        msd = FindObjectOfType<MeshSimilarityDebugger>();
#endif
    }

    public void ApplySelected(bool editTime, bool alsoCompute)
    {

        if (presetToApply == Preset.None) return; // <— Manuel mod: hiç dokunma
        if (!msd) TryResolveMSD();
        if (!msd) return;

        switch (presetToApply)
        {
            case Preset.Task3:
                Apply2D_Hole(msd);
                break;
            case Preset.Task1:
                Apply3D_Global(msd);
                break;
            case Preset.Task2:
                ApplyRotationOnly(msd);
                break;
        }

#if UNITY_EDITOR
        if (editTime)
        {
            Undo.RecordObject(msd, "Apply MSD Preset");
            EditorUtility.SetDirty(msd);
            var scene = msd.gameObject.scene;
            if (scene.IsValid()) EditorSceneManager.MarkSceneDirty(scene);
        }
#endif

        if (alsoCompute && (!Application.isPlaying || (Application.isPlaying && msd != null)))
        {
            msd.ComputeAll();
        }
    }


    // -------- Presetler --------
    public static void ApplyRotationOnly(MeshSimilarityDebugger msd)
    {

        // Önce Global3D tabanını uygula:
        Apply3D_Global(msd);
   
        // rotationReferenceGlobal artık önemli değil; mesh-tabanlı ölçüm kullanılıyor.
        // Global3D + rotasyon füzyonu:
        msd.addChildRotationPenaltyToGlobal = true;                 // <<< YENİ: füzyonu aç
        msd.useRotationPenalty = true;                              // rotSim hesapla
        msd.rotationReferenceGlobal = MeshSimilarityDebugger.RotationReference.RootRelative;

        // Rotasyon benzerliği eğrisi (RotationSimilarityFromTheta parametreleri)
        msd.rotCapDeg = 20f;       // açı tavanı (daha yumuşak kıvrım için 15-30)
        msd.rotGamma = 2.2f;       // düşüş eğrisi (1.5-3 arası deneyin)
        msd.rotEmphasis = 2.0f;    // keskinlik (1.2-3 arası)

        // Global ceza katkısı (mesafeye eklenen pay): lambda * penalty01 * diag
        msd.rotPenaltyGlobalLambda = 1f; // 0.3-1.0 bandı pratikte mantıklı
        msd.rotPenaltyChildMultiplier = 10f;
   
        // Per-child pozisyon vs. kapalı; istenirse ayrı bir presetle açılabilir.
        msd.usePositionPenalty = false;

        // Not: perChildMode GLOBAL kalacak (false). Eşleştirme, ComputeChildRotationPenaltyAvg01 içinde
        // index-sırası ile yapılır; mevcut mantık birebir korunur.
    }


    public static void Apply2D_Hole(MeshSimilarityDebugger msd)
    {
        // --- Genel / Örnekleme ---
        msd.perChildMode = false;
        msd.poseInvariant = true;
        msd.samplesPerMesh = 12000;
        msd.randomSeed = 12345;

        msd.emphasizeEdges = true;
        msd.edgePortion = 1f;               // sadece boundary
        msd.holeAwareMatching = true;
        msd.holeEdgeWeight = 4.5f;
        msd.hausdorffBoundaryOnly = true;

        // --- Hizalama / 2D ---
        msd.centerAlignByBounds = true;
        msd.projectPlanar = false;
        msd.forcePlanarFor2D = true;

        // --- Metrik / Hausdorff ---
        msd.metric = SimilarityMetric.TrimmedHausdorff;
        msd.hausdorffPercentile = 0.998f;
        msd.matchDeltaFrac = 0.003f;
        msd.useTrimmed = false;
        msd.useCapped = false;
        msd.capAtDiagFrac = 0.10f;
        msd.distanceExponent = 3.0f;

        // --- Penalty’ler kapalı (sadece outer’a etki edenler) ---
        msd.useNormalPenalty = false;
        msd.useRotationPenalty = false;
        msd.useScalePenalty = false;

        msd.hausdorffAggregation = MeshSimilarityDebugger.HausdorffAgg.MaxOfDirs;
      
        msd.childAggregation = MeshSimilarityDebugger.ChildAgg.Mean;
        msd.simSharpnessPow = 2.2f;

        // --- Yeni: Hole W/H + Count ---
        msd.useHoleWHPenalty = true;
        msd.holeWHPenaltyLambda = 1.2f;
        msd.holeWHExp = 2.4f;
        msd.holeWHTopFrac = 0.85f;
        msd.holeWHRankWeighted = true;

        msd.useHoleCountOnly = true;
        msd.holeCountPenaltyLambda = 0.8f;
    }


    public static void Apply3D_Global(MeshSimilarityDebugger msd)
    {
        // --- Mod / Örnekleme ---
        msd.perChildMode = false;
        msd.samplesPerMesh = 7000;          // 5000 → 7000: istikrar için hafif artış
        msd.poseInvariant = true;
        msd.randomSeed = 12345;

        // Yüzey + hacim
        msd.emphasizeEdges = false;
        msd.edgePortion = 0.35f;

        // 3B
        msd.holeAwareMatching = false;
        msd.hausdorffBoundaryOnly = false;

        // Hizalama
        msd.centerAlignByBounds = true;
        msd.projectPlanar = false;
        msd.forcePlanarFor2D = false;

        // --- Metrik: biçim odaklı Chamfer ---
        msd.metric = SimilarityMetric.ChamferMean;
        msd.useTrimmed = false;     // küçük outlier budaması
        msd.trimTopPercent = 0.015f;   // %1.5
        msd.hausdorffAggregation = MeshSimilarityDebugger.HausdorffAgg.MeanOfDirs;
        msd.distanceExponent = 1.20f;    // 1.3 → 1.2

        // Cap: tekil uzun farkları sınırlı tut
        msd.useCapped = false;            // kapalıydı → aç
        msd.capAtDiagFrac = 0.12f;           // 0.08 → 0.12

        // Eşleşme toleransı (istatistik)
        msd.matchDeltaFrac = 0.012f;

        // Ölçek cezası: biçime öncelik
        msd.useScalePenalty = true;
        msd.scalePenaltyLambda = 1f;      // 1.0 → 0.6
        msd.axisPenaltyWeights = new Vector3(1f, 1f, 1f);

    

        // Diğer cezalar kapalı
        msd.useNormalPenalty = false;
        msd.useRotationPenalty = false;

        // Toplama & keskinlik
        msd.childAggregation = MeshSimilarityDebugger.ChildAgg.Mean;
   
        msd.simSharpnessPow = 1.50f;       // 1.25 → 1.15 (yüksek benzerlikleri sıkıştırma)
    }

    // Runtime: sahne yüklendikten hemen sonra uygula (domain/scene reload olsa da)
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void ApplyForAllInScene_AfterSceneLoad()
    {
        var appliers = Object.FindObjectsOfType<MSDPresetApplier>(true);
        foreach (var a in appliers)
        {
            if (a && a.applyOnEnterPlay)
                a.ApplySelected(editTime: false, alsoCompute: false);
        }
    }
}

#if UNITY_EDITOR
/// Editör hook'ları: Play'e girerken/çıkarken otomatik uygulama
[InitializeOnLoad]
public static class MSDPresetPlayHooks
{
    static MSDPresetPlayHooks()
    {
        EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
    }

    static void OnPlayModeStateChanged(PlayModeStateChange state)
    {
        // Play'e girmeden hemen önce: Edit önizleme istendiyse uygula
        if (state == PlayModeStateChange.ExitingEditMode)
        {
            var appliers = Resources.FindObjectsOfTypeAll<MSDPresetApplier>()
                .Where(a => a && a.gameObject.scene.IsValid());

            foreach (var a in appliers)
            {
                if (a.applyInEditPreview)
                    a.ApplySelected(editTime: true, alsoCompute: a.computeInEditPreview);
            }

            SceneView.RepaintAll();
        }

        // Play'e girildi: ek güvenlik (reload davranışları için)
        if (state == PlayModeStateChange.EnteredPlayMode)
        {
            var appliers = Object.FindObjectsOfType<MSDPresetApplier>(true);
            foreach (var a in appliers)
            {
                if (a.applyOnEnterPlay)
                    a.ApplySelected(editTime: false, alsoCompute: false);
            }
        }
    }
}
#endif
