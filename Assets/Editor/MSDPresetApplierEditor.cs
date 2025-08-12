// MSDPresetApplierEditor.cs
#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(MSDPresetApplier))]
public class MSDPresetApplierEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var applier = (MSDPresetApplier)target;
        GUILayout.Space(8);
        EditorGUILayout.LabelField("Edit Mode Quick Actions", EditorStyles.boldLabel);

        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Apply Now (Edit)"))
            {
                applier.ApplySelected(editTime: true, alsoCompute: false);
            }
            if (GUILayout.Button("Apply + ComputeAll (Edit)"))
            {
                applier.ApplySelected(editTime: true, alsoCompute: true);
            }
        }

        if (applier.msd == null)
        {
            EditorGUILayout.HelpBox(
                "msd alaný boþ. Objede veya sahnede MeshSimilarityDebugger otomatik aranýr.\n" +
                "Birden fazla varsa hangisini kullandýðýný elle baðlaman önerilir.",
                MessageType.Info);
        }
    }
}
#endif
