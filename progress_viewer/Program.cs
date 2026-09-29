using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Windows.Forms;

namespace ProgressViewer;

internal static class Program
{
    [STAThread]
    private static void Main(string[] args)
    {
        ApplicationConfiguration.Initialize();

        string? progressPath = null;
        if (args.Length > 0 && !string.IsNullOrWhiteSpace(args[0]))
        {
            progressPath = args[0];
        }

        if (string.IsNullOrWhiteSpace(progressPath) || !File.Exists(progressPath))
        {
            progressPath = FindDefaultProgressPath();
        }

        if (string.IsNullOrWhiteSpace(progressPath) || !File.Exists(progressPath))
        {
            using var dialog = new OpenFileDialog
            {
                Title = "progress.json を選択",
                Filter = "JSON files (*.json)|*.json|All files (*.*)|*.*",
                CheckFileExists = true
            };
            string? initial = FindDefaultProgressDirectory();
            if (!string.IsNullOrEmpty(initial))
            {
                dialog.InitialDirectory = initial;
                dialog.FileName = "progress.json";
            }
            if (dialog.ShowDialog() == DialogResult.OK)
            {
                progressPath = dialog.FileName;
            }
        }

        if (string.IsNullOrWhiteSpace(progressPath))
        {
            MessageBox.Show(
                "進捗ファイルが選択されていません。",
                "3方向合成 進捗ビューア",
                MessageBoxButtons.OK,
                MessageBoxIcon.Information);
            return;
        }

        Application.Run(new ProgressViewerForm(progressPath));
    }

    private static string? FindDefaultProgressPath()
    {
        foreach (var path in DefaultProgressCandidates())
        {
            if (File.Exists(path))
            {
                return path;
            }
        }
        return null;
    }

    private static string? FindDefaultProgressDirectory()
    {
        foreach (var path in DefaultProgressCandidates())
        {
            string? dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && Directory.Exists(dir))
            {
                return dir;
            }
        }
        return null;
    }

    private static string[] DefaultProgressCandidates()
    {
        string exeDir = AppContext.BaseDirectory;
        string cwd = Directory.GetCurrentDirectory();
        return new[]
        {
            Path.Combine(cwd, "progress", "progress.json"),
            Path.Combine(exeDir, "progress", "progress.json"),
            Path.GetFullPath(Path.Combine(exeDir, "..", "..", "..", "..", "progress", "progress.json")),
        };
    }
}

internal sealed class ProgressViewerForm : Form
{
    private readonly string _progressPath;
    private readonly System.Windows.Forms.Timer _timer;
    private readonly JsonSerializerOptions _jsonOptions;

    private readonly ProgressBar _overallBar;
    private readonly Label _overallLabel;
    private readonly Label _etaLabel;
    private readonly ProgressBar _stepBar;
    private readonly Label _stepLabel;
    private readonly ListView _stepsView;
    private readonly TextBox _detailsBox;
    private readonly Label _fileLabel;
    private readonly Label _statusLabel;

    public ProgressViewerForm(string progressPath)
    {
        _progressPath = Path.GetFullPath(progressPath);
        Text = "3方向合成 進捗ビューア";
        Width = 920;
        Height = 742;
        MinimumSize = new Size(720, 540);
        StartPosition = FormStartPosition.CenterScreen;

        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            AllowTrailingCommas = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
        };

        _fileLabel = new Label
        {
            Text = $"ファイル: {_progressPath}",
            AutoSize = false,
            Dock = DockStyle.Top,
            Height = 28,
            Padding = new Padding(8, 6, 8, 0),
        };

        _overallLabel = new Label
        {
            Text = "全体進捗: 0%",
            AutoSize = false,
            Dock = DockStyle.Top,
            Height = 22,
            Padding = new Padding(8, 0, 8, 0),
        };

        _overallBar = new ProgressBar
        {
            Dock = DockStyle.Top,
            Height = 18,
            Minimum = 0,
            Maximum = 100
        };

        _etaLabel = new Label
        {
            Text = "経過: --    残り: 算出中    終了予定: --",
            AutoSize = false,
            Dock = DockStyle.Top,
            Height = 22,
            Padding = new Padding(8, 0, 8, 0),
        };

        _stepLabel = new Label
        {
            Text = "現在のステップ: （なし）",
            AutoSize = false,
            Dock = DockStyle.Top,
            Height = 22,
            Padding = new Padding(8, 0, 8, 0),
        };

        _stepBar = new ProgressBar
        {
            Dock = DockStyle.Top,
            Height = 18,
            Minimum = 0,
            Maximum = 100
        };

        _stepsView = new ListView
        {
            Dock = DockStyle.Fill,
            View = View.Details,
            FullRowSelect = true,
            GridLines = true,
        };
        _stepsView.Columns.Add("ID", 40);
        _stepsView.Columns.Add("ステップ", 140);
        _stepsView.Columns.Add("状態", 90);
        _stepsView.Columns.Add("進捗", 70);
        _stepsView.Columns.Add("開始", 150);
        _stepsView.Columns.Add("終了", 150);
        _stepsView.Columns.Add("詳細", 220);

        _detailsBox = new TextBox
        {
            Dock = DockStyle.Bottom,
            Height = 160,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            Font = new Font(FontFamily.GenericMonospace, 9f),
        };

        _statusLabel = new Label
        {
            Text = "更新待ち...",
            AutoSize = false,
            Dock = DockStyle.Bottom,
            Height = 22,
            Padding = new Padding(8, 2, 8, 0),
        };

        var topPanel = new Panel { Dock = DockStyle.Top, Height = 132 };
        topPanel.Controls.Add(_stepBar);
        topPanel.Controls.Add(_stepLabel);
        topPanel.Controls.Add(_etaLabel);
        topPanel.Controls.Add(_overallBar);
        topPanel.Controls.Add(_overallLabel);
        topPanel.Controls.Add(_fileLabel);

        Controls.Add(_stepsView);
        Controls.Add(_detailsBox);
        Controls.Add(_statusLabel);
        Controls.Add(topPanel);

        _timer = new System.Windows.Forms.Timer { Interval = 500 };
        _timer.Tick += (_, _) => LoadProgress();
        _timer.Start();

        FormClosed += (_, _) => _timer.Stop();
        Load += (_, _) => LoadProgress();
    }

    private void LoadProgress()
    {
        try
        {
            if (!File.Exists(_progressPath))
            {
                _statusLabel.Text = "progress.json が見つかりません。作成待ちです。";
                return;
            }

            string json;
            using (var fs = new FileStream(
                _progressPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete))
            using (var sr = new StreamReader(fs))
            {
                json = sr.ReadToEnd();
            }

            var progress = JsonSerializer.Deserialize<ProcessProgress>(json, _jsonOptions);
            if (progress == null)
            {
                _statusLabel.Text = "progress.json が空、または不正です。";
                return;
            }

            UpdateUi(progress);
            var lastWrite = File.GetLastWriteTime(_progressPath);
            _statusLabel.Text = $"最終更新: {lastWrite:yyyy-MM-dd HH:mm:ss}";
        }
        catch (IOException)
        {
            _statusLabel.Text = "ファイル書き込み中です。再試行します...";
        }
        catch (JsonException ex)
        {
            _statusLabel.Text = $"JSON 解析エラー: {ex.Message}";
        }
        catch (Exception ex)
        {
            _statusLabel.Text = $"予期しないエラー: {ex.Message}";
        }
    }

    private void UpdateUi(ProcessProgress progress)
    {
        var overall = ClampPercent(progress.overall_progress);
        _overallBar.Value = overall;
        _overallLabel.Text = $"全体進捗: {overall}%    id={progress.process_id ?? "-"}";
        _etaLabel.Text = FormatEtaLine(progress);

        var steps = progress.steps ?? Array.Empty<StepInfo>();
        StepInfo? current = null;

        if (progress.current_step.HasValue)
        {
            current = steps.FirstOrDefault(s => s.step_id == progress.current_step.Value);
        }
        if (current == null)
        {
            current = steps.FirstOrDefault(s => string.Equals(s.status, "in_progress", StringComparison.OrdinalIgnoreCase));
        }

        if (current != null)
        {
            var stepProgress = ClampPercent(current.progress);
            _stepBar.Value = stepProgress;
            _stepLabel.Text = $"現在のステップ: {current.name_ja ?? current.name_en ?? "(不明)"} ({stepProgress}%)";
            _detailsBox.Text = FormatDetails(current.details);
        }
        else
        {
            _stepBar.Value = 0;
            bool allDone = steps.Length > 0 &&
                steps.All(s => s.status is "completed" or "skipped");
            _stepLabel.Text = allDone ? "現在のステップ: 全ステップ完了" : "現在のステップ: （なし）";
            _detailsBox.Text = string.Empty;
        }

        _stepsView.BeginUpdate();
        _stepsView.Items.Clear();
        foreach (var step in steps.OrderBy(s => s.step_id))
        {
            var item = new ListViewItem(step.step_id.ToString());
            item.SubItems.Add(step.name_ja ?? step.name_en ?? "(不明)");
            item.SubItems.Add(StatusJa(step.status));
            item.SubItems.Add($"{ClampPercent(step.progress)}%");
            item.SubItems.Add(FormatTime(step.start_time));
            item.SubItems.Add(FormatTime(step.end_time));
            item.SubItems.Add(SummarizeDetails(step.details));
            item.BackColor = StatusColor(step.status, current != null && step.step_id == current.step_id);
            _stepsView.Items.Add(item);
        }
        _stepsView.EndUpdate();
    }

    private static Color StatusColor(string? status, bool isCurrent)
    {
        if (isCurrent)
        {
            return Color.LightYellow;
        }
        return status switch
        {
            "completed" => Color.Honeydew,
            "error" => Color.MistyRose,
            "skipped" => Color.Gainsboro,
            "in_progress" => Color.LightYellow,
            _ => Color.White,
        };
    }

    private static string StatusJa(string? status) => status switch
    {
        "pending" => "待機",
        "in_progress" => "実行中",
        "completed" => "完了",
        "error" => "エラー",
        "skipped" => "スキップ",
        _ => status ?? "",
    };

    private static string FormatEtaLine(ProcessProgress progress)
    {
        string elapsed = FormatDuration(progress.elapsed_time_seconds);
        string remaining;
        string eta;
        bool hasError = (progress.steps ?? Array.Empty<StepInfo>()).Any(
            s => string.Equals(s.status, "error", StringComparison.OrdinalIgnoreCase));
        if (progress.estimated_remaining_seconds is double rem)
        {
            if (rem <= 0.05)
            {
                remaining = "0秒";
                eta = "完了";
            }
            else
            {
                remaining = FormatDuration(rem);
                eta = FormatDateTime(progress.estimated_end_time);
                if (string.IsNullOrEmpty(eta))
                {
                    eta = "--";
                }
            }
        }
        else
        {
            remaining = hasError ? "--" : "算出中";
            eta = "--";
        }
        return $"経過: {elapsed}    残り: {remaining}    終了予定: {eta}";
    }

    private static string FormatDuration(double seconds)
    {
        if (double.IsNaN(seconds) || double.IsInfinity(seconds) || seconds < 0)
        {
            return "--";
        }
        var span = TimeSpan.FromSeconds(seconds);
        if (span.TotalHours >= 1)
        {
            return $"{(int)span.TotalHours}時間{span.Minutes}分";
        }
        if (span.TotalMinutes >= 1)
        {
            return $"{span.Minutes}分{span.Seconds}秒";
        }
        return $"{Math.Max(0, (int)Math.Round(span.TotalSeconds))}秒";
    }

    private static string FormatTime(string? iso)
    {
        if (string.IsNullOrWhiteSpace(iso))
        {
            return "";
        }
        if (DateTime.TryParse(iso, out var dt))
        {
            return dt.ToString("HH:mm:ss");
        }
        return iso;
    }

    private static string FormatDateTime(string? iso)
    {
        if (string.IsNullOrWhiteSpace(iso))
        {
            return "";
        }
        if (DateTime.TryParse(iso, out var dt))
        {
            return dt.ToString("yyyy-MM-dd HH:mm:ss");
        }
        return iso;
    }

    private static int ClampPercent(double value)
    {
        if (double.IsNaN(value) || double.IsInfinity(value)) return 0;
        if (value < 0) return 0;
        if (value > 100) return 100;
        return (int)Math.Round(value, MidpointRounding.AwayFromZero);
    }

    private static string SummarizeDetails(JsonElement? details)
    {
        if (details == null || !details.HasValue)
        {
            return "";
        }
        var el = details.Value;
        if (el.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
        {
            return "";
        }
        if (el.ValueKind != JsonValueKind.Object)
        {
            return el.ToString();
        }

        var parts = new List<string>();
        foreach (var key in new[]
        {
            "run_id", "phase", "current_frame", "total_frames",
            "run_index", "n_runs", "skip_reason", "error_message",
        })
        {
            if (el.TryGetProperty(key, out var value) &&
                value.ValueKind is not JsonValueKind.Null and not JsonValueKind.Undefined)
            {
                parts.Add($"{key}={TrimJson(value)}");
            }
        }
        return string.Join("  ", parts);
    }

    private static string FormatDetails(JsonElement? details)
    {
        if (details == null || !details.HasValue)
        {
            return string.Empty;
        }

        try
        {
            var el = details.Value;
            if (el.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
            {
                return string.Empty;
            }
            string summary = SummarizeDetails(details);
            using var doc = JsonDocument.Parse(el.GetRawText());
            string pretty = JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true });
            if (string.IsNullOrEmpty(summary))
            {
                return pretty;
            }
            return summary + Environment.NewLine + Environment.NewLine + pretty;
        }
        catch
        {
            return details.Value.ToString() ?? string.Empty;
        }
    }

    private static string TrimJson(JsonElement value)
    {
        string raw = value.ToString();
        if (raw.Length >= 2 && raw[0] == '"' && raw[^1] == '"')
        {
            return raw[1..^1];
        }
        return raw;
    }
}

internal sealed class ProcessProgress
{
    [JsonPropertyName("process_id")]
    public string? process_id { get; set; }

    [JsonPropertyName("overall_progress")]
    public double overall_progress { get; set; }

    [JsonPropertyName("current_step")]
    public int? current_step { get; set; }

    [JsonPropertyName("steps")]
    public StepInfo[]? steps { get; set; }

    [JsonPropertyName("start_time")]
    public string? start_time { get; set; }

    [JsonPropertyName("elapsed_time_seconds")]
    public double elapsed_time_seconds { get; set; }

    [JsonPropertyName("estimated_remaining_seconds")]
    public double? estimated_remaining_seconds { get; set; }

    [JsonPropertyName("estimated_end_time")]
    public string? estimated_end_time { get; set; }
}

internal sealed class StepInfo
{
    [JsonPropertyName("step_id")]
    public int step_id { get; set; }

    [JsonPropertyName("name_ja")]
    public string? name_ja { get; set; }

    [JsonPropertyName("name_en")]
    public string? name_en { get; set; }

    [JsonPropertyName("status")]
    public string? status { get; set; }

    [JsonPropertyName("progress")]
    public double progress { get; set; }

    [JsonPropertyName("start_time")]
    public string? start_time { get; set; }

    [JsonPropertyName("end_time")]
    public string? end_time { get; set; }

    [JsonPropertyName("details")]
    public JsonElement? details { get; set; }
}
