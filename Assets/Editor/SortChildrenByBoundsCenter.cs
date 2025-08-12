using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;

public class SortChildrenByBoundsCenter : MonoBehaviour
{
    [MenuItem("Tools/Sort Children By Bounds Center (XYZ)")]
    private static void SortSelectedChildren()
    {
        // Kullanıcının sahnede seçtiği GameObject
        GameObject selected = Selection.activeGameObject;
        if (selected == null)
        {
            Debug.LogError("Lütfen bir GameObject seçin.");
            return;
        }

        if (selected.transform.childCount == 0)
        {
            Debug.LogWarning("Seçilen nesnenin çocuğu yok.");
            return;
        }

        // Alt çocukların listesini al
        List<Transform> children = new List<Transform>();
        foreach (Transform child in selected.transform)
        {
            children.Add(child);
        }

        // Bounds.center değerine göre sırala (önce X, sonra Y, sonra Z)
        children = children.OrderBy(c => GetRendererCenter(c).x)
                           .ThenBy(c => GetRendererCenter(c).y)
                           .ThenBy(c => GetRendererCenter(c).z)
                           .ToList();

        // Yeni sıralamayı sahnede uygula
        for (int i = 0; i < children.Count; i++)
        {
            children[i].SetSiblingIndex(i);
        }

        Debug.Log($"{selected.name} içindeki çocuklar Bounds.Center (X→Y→Z) sırasına göre sıralandı.");
    }

    private static Vector3 GetRendererCenter(Transform t)
    {
        Renderer rend = t.GetComponentInChildren<Renderer>();
        if (rend != null)
        {
            return rend.bounds.center;
        }
        else
        {
            // Renderer yoksa sadece localPosition kullan
            return t.position;
        }
    }
}
