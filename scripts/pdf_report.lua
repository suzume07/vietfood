-- Convert the report's Markdown constructs into print-friendly LaTeX.
local function latex(text)
  return pandoc.write(pandoc.Pandoc({pandoc.Plain({pandoc.Str(text)})}), 'latex'):gsub('\n$', '')
end

function Code(code)
  -- Escape Windows backslashes as text, including inside table/minipage
  -- arguments where URL's verbatim scanner cannot operate reliably.
  local replacements = {['\\']='\\textbackslash{}', ['{']='\\{', ['}']='\\}', ['%']='\\%', ['&']='\\&', ['#']='\\#', ['_']='\\_', ['$']='\\$', ['~']='\\textasciitilde{}', ['^']='\\textasciicircum{}'}
  local parts = {}
  for _, value in utf8.codes(code.text) do
    local char = utf8.char(value)
    parts[#parts + 1] = (replacements[char] or char) .. '\\allowbreak{}'
  end
  return pandoc.RawInline('latex', '\\texttt{' .. table.concat(parts) .. '}')
end

function CodeBlock(code)
  if code.classes:includes('mermaid') then
    local labels = {}
    for _, label in code.text:gmatch('([%w_]+)%[([^%]]+)%]') do
      labels[#labels + 1] = label
    end
    local columns = #labels > 6 and 4 or 3
    local width = columns == 4 and '3.25cm' or '4.1cm'
    local spacing = columns == 4 and 3.8 or 4.9
    local parts = {'\\begin{center}\n\\begin{tikzpicture}[every node/.style={draw=blue!55!black,fill=blue!4,rounded corners=2pt,text width=' .. width .. ',minimum height=1.55cm,align=center,font=\\small},>=stealth]'}
    for i, label in ipairs(labels) do
      local row = math.floor((i - 1) / columns)
      local col = (i - 1) % columns
      if row % 2 == 1 then col = columns - 1 - col end
      parts[#parts + 1] = string.format('\\node (n%d) at (%.2f,%.2f) {%s};', i, col * spacing, -row * 2.2, latex(label))
    end
    for i = 1, #labels - 1 do
      parts[#parts + 1] = string.format('\\draw[->,thick,blue!55!black] (n%d) -- (n%d);', i, i + 1)
    end
    parts[#parts + 1] = '\\end{tikzpicture}\n\\end{center}'
    return pandoc.RawBlock('latex', table.concat(parts, '\n'))
  end
  return pandoc.RawBlock('latex', '\\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\\footnotesize,baselinestretch=1]\n' .. code.text .. '\n\\end{Verbatim}')
end

function Para(para)
  if #para.content == 1 and para.content[1].t == 'Image' then
    local image = para.content[1]
    if image.src:find('test_error_examples.png', 1, true) then
      -- Original 1800 x 3330 px @ 150 dpi. Three landscape panels keep
      -- the six example rows readable without changing the source image.
      local parts = {}
      for i = 1, 3 do
        local bottom = (3 - i) * 532.8
        local top = bottom + 532.8
        parts[#parts + 1] = string.format('\\begin{landscape}\n\\centering\n\\includegraphics[viewport=0 %.1f 864 %.1f,clip,width=\\linewidth,height=0.86\\textheight,keepaspectratio]{%s}\n\\par\\medskip\n{\\small\\textbf{Hình 5%s.} Mẫu %d–%d: bên trái là GT, bên phải là dự đoán tại confidence 0,5936.}\n\\end{landscape}', bottom, top, image.src, string.char(96+i), (i-1)*2+1, i*2)
      end
      return pandoc.RawBlock('latex', table.concat(parts, '\n'))
    end
    return pandoc.RawBlock('latex', '\\begin{center}\n\\includegraphics[width=\\linewidth,height=0.76\\textheight,keepaspectratio]{' .. image.src .. '}\n\\end{center}')
  end
end

function Table(tbl)
  -- Give prose and filenames more space than short numeric cells.
  local columns = #tbl.colspecs
  local first_row = tbl.head.rows[1]
  if first_row then
    local headers = {}
    for _, cell in ipairs(first_row.cells) do
      headers[#headers + 1] = pandoc.utils.stringify(cell.contents)
    end
    local weights
    if columns == 7 then
      weights = {0.19, 0.135, 0.135, 0.115, 0.115, 0.10, 0.21}
    elseif columns == 6 and headers[2] == 'File' then
      weights = {0.07, 0.65, 0.07, 0.07, 0.07, 0.07}
    elseif columns == 2 then
      weights = {0.32, 0.68}
    elseif columns == 3 and headers[1] ~= 'Tập' then
      weights = {0.23, 0.34, 0.43}
    end
    if weights then
      for i, w in ipairs(weights) do tbl.colspecs[i][2] = w end
    end
  end
  return {pandoc.RawBlock('latex', '\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.12}'), tbl, pandoc.RawBlock('latex', '\\endgroup')}
end

function Pandoc(doc)
  local output = pandoc.List()
  local skip_toc = false
  local intro = true
  for _, block in ipairs(doc.blocks) do
    if block.t == 'Header' then
      local title = pandoc.utils.stringify(block.content)
      if block.level == 1 and intro then
        -- Report title and author metadata are rendered on a dedicated cover.
      elseif title == 'MỤC LỤC' then
        output:insert(pandoc.RawBlock('latex', '\\clearpage\\tableofcontents\\clearpage'))
        skip_toc = true
      else
        skip_toc = false
        intro = false
        block.level = math.max(1, block.level - 1)
        output:insert(block)
      end
    elseif skip_toc then
      -- Replace the hand-written list with the actual LaTeX page-numbered TOC.
    elseif intro and block.t == 'Para' and pandoc.utils.stringify(block.content):find('Lĩnh vực:', 1, true) then
      -- This metadata is on the cover, with the values read from Markdown.
    else
      output:insert(block)
    end
  end
  doc.blocks = output
  return doc
end
