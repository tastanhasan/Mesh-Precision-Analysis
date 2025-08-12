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

    public enum Preset { None, Hole2D, Global3D, RotationOnly }   // <— None eklendi

    [Header("Preset")]
    public Preset presetToApply = Preset.Hole2D;

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
            case Preset.Hole2D:
                Apply2D_Hole(msd);
                break;
            case Preset.Global3D:
                Apply3D_Global(msd);
                break;
            case Preset.RotationOnly:
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
        // --- Mod ---
        msd.perChildMode = true;
        msd.childCenterAlignByBounds = false; // pozisyon farkı dahil
        msd.poseInvariant = false;
        msd.samplesPerMesh = 1000;
        msd.randomSeed = 12345;

        // --- Rotasyon penalty ---
        msd.useNormalPenalty = false;
        msd.useRotationPenalty = true;
        msd.rotationReference = MeshSimilarityDebugger.RotationReference.World;
        msd.rotMode = MeshSimilarityDebugger.RotationPenaltyMode.LinearBlend;

        msd.rotWeight = 0.7f;   // rotasyon etkisini orta seviyede tut
        msd.rotCapDeg = 35f;    // tolerans
        msd.rotGamma = 1.8f;   // eğri dikliği
        msd.rotEmphasis = 1.5f;   // etki şiddeti

        // --- Pozisyon penalty ---
        msd.usePositionPenalty = true;
        msd.positionReference = MeshSimilarityDebugger.PositionReference.World;
        msd.posCap = 0.10f; // sahne ölçeğine göre ~10 cm
        msd.posGamma = 1.4f;
        msd.posEmphasis = 1.2f;

        // Rot+Poz birleşimi
        msd.poseCombine = MeshSimilarityDebugger.PoseCombine.LinearBlend;
        msd.poseBlendWeight = 0.3f; // pozisyon %30 etkili

        // --- Toplama / keskinlik ---
        msd.childAggregation = MeshSimilarityDebugger.ChildAgg.Mean;
        msd.globalAggregation = MeshSimilarityDebugger.GlobalAgg.Mean;
        msd.simSharpnessPow = 1.5f; // orta keskinlik

        // --- Şekil etkisi minimum ---
        msd.metric = SimilarityMetric.TrimmedHausdorff;
        msd.hausdorffPercentile = 0.999f;
        msd.useTrimmed = false;
        msd.useCapped = false;
        msd.matchDeltaFrac = 0.005f;
        msd.distanceExponent = 1.5f;

        // Diğerlerini kapalı tut
        msd.emphasizeEdges = false;
        msd.holeAwareMatching = false;
        msd.hausdorffBoundaryOnly = false;
        msd.centerAlignByBounds = true;
        msd.projectPlanar = false;
        msd.forcePlanarFor2D = false;
        msd.useScalePenalty = false;

        msd.hausdorffAggregation = MeshSimilarityDebugger.HausdorffAgg.MaxOfDirs;
        msd.diagonalNorm = MeshSimilarityDebugger.DiagNorm.Min;
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
        msd.diagonalNorm = MeshSimilarityDebugger.DiagNorm.Min;
        msd.globalAggregation = MeshSimilarityDebugger.GlobalAgg.Mean;
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
        // Mod
        msd.perChildMode = false;

        // Örnekleme
        msd.samplesPerMesh = 5000;
        msd.poseInvariant = true;
        msd.randomSeed = 12345;

        // Kenar vurgusu
        msd.emphasizeEdges = false;
        msd.edgePortion = 0.35f;

        // Delik duyarlılığı kapalı (3B)
        msd.holeAwareMatching = false;
        msd.hausdorffBoundaryOnly = false;

        // Hizalama / 3B
        msd.centerAlignByBounds = true;
        msd.projectPlanar = false;
        msd.forcePlanarFor2D = false;

        // Metrik
        msd.metric = SimilarityMetric.TrimmedHausdorff; // global enum
        msd.hausdorffPercentile = 0.999f;

        // Dayanıklılık
        msd.useTrimmed = false;
        msd.trimTopPercent = 0.05f;
        msd.useCapped = false;
        msd.capAtDiagFrac = 0.10f;
        msd.matchDeltaFrac = 0.005f;
        msd.distanceExponent = 4f;

        // Ölçek cezası
        msd.useScalePenalty = true;
        msd.scalePenaltyLambda = 2f;
        msd.axisPenaltyWeights = new Vector3(1f, 1f, 1f);

        // Hausdorff yön ve diag norm
        msd.hausdorffAggregation = MeshSimilarityDebugger.HausdorffAgg.MaxOfDirs;
        msd.diagonalNorm = MeshSimilarityDebugger.DiagNorm.Min;

        // (Per-Child ayarları burada etkisiz; tutarlı dursun)
        msd.useNormalPenalty = true;
        msd.normalPenaltyLambda = 2f;
        msd.useRotationPenalty = false;

        msd.childAggregation = MeshSimilarityDebugger.ChildAgg.Min;
        msd.globalAggregation = MeshSimilarityDebugger.GlobalAgg.Mean;
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
